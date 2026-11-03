"""Native ExLlamaV3 test; never execute code downloaded with the model."""
from pathlib import Path
import argparse, json, os, sys, time, resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))

ap=argparse.ArgumentParser(description="Short-prompt gfx1103 reproduction; no service or system changes")
ap.add_argument('--repo',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output',required=True)
ap.add_argument('--mode',choices=['off','on'],required=True);ap.add_argument('--draft-tokens',type=int,choices=[2,3],default=2)
a=ap.parse_args()
REPO=Path(a.repo).resolve(); MODEL=str(Path(a.model).resolve()); OUT=Path(a.output).resolve()
OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(REPO))
settings={'ndt':a.draft_tokens,'threads':8,'kv_bits':0,'dynamic':False}
a.smoke=False; a.profile_smoke=False; a.preset_file=None
# Child-only environment; matches the validated fixed-draft profiles.
ENV={'EXL3_RDNA3_GEMM': '1', 'EXL3_RDNA3_ATTN': '1', 'EXL3_HIP_MGEMM': '1', 'EXL3_FUSE_ACT': '1', 'EXL3_FUSE_GNORM': '1', 'EXL3_GDN_REG': '1', 'EXL3_FUSE_NORM_HAD': '1', 'EXL3_RESID_DEFER': '1', 'EXL3_ABL_FUSE': '1', 'EXL3_PRESAMPLE': '1', 'EXL3_REWIND_CACHE': '1', 'EXL3_ACT_EPI': '1', 'EXL3_ABLATION': '1', 'EXL3_GDN_MK': '0', 'EXL3_NOGRAPH': 'mlp,gdn', 'EXL3_RDNA3_TARGET_BLOCKS': '36', 'EXL3_RDNA3_ATTN_SPLIT_MULT': '16'}
ENV['ROCBLAS_TENSILE_LIBPATH']='/opt/rocm/lib/rocblas/library'
ENV['PYTORCH_ROCM_ARCH']='gfx1103'
if os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
    raise SystemExit('Use a clean native gfx1103 environment without HSA_OVERRIDE_GFX_VERSION')
os.environ.update(ENV)
# Preserve the original 96 C CPU/GPU cutoff. Missing thermal monitoring aborts.
import threading

def watch_temperature():
    try:
        sensors=[]
        for h in Path('/sys/class/hwmon').glob('hwmon*'):
            if (h/'name').read_text().strip() in ('amdgpu','k10temp'):
                sensors.extend(h.glob('temp*_input'))
        if not sensors: raise RuntimeError('No CPU/GPU thermal sensors')
        with (OUT/'thermal.jsonl').open('w') as log:
            while True:
                values={str(p):float(p.read_text())/1000 for p in sensors}
                log.write(json.dumps({'time':time.time(),'temperatures_C':values})+'\n'); log.flush()
                if max(values.values())>=96: raise RuntimeError('96 C cutoff reached')
                time.sleep(.5)
    except Exception as e:
        print('THERMAL_GUARD:',e,flush=True); os._exit(96)
threading.Thread(target=watch_temperature,daemon=True).start()

def status(stage,**fields):
    row={'time':time.time(),'stage':stage,**fields}
    (OUT/'status.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)

status('import')
import torch
import importlib.util
extension_spec=importlib.util.find_spec('exllamav3_ext')
if not extension_spec or not extension_spec.origin or Path(extension_spec.origin).parent!=REPO or not extension_spec.origin.endswith('.so'):
    raise SystemExit('First build the extension in the requested checkout with build_ext --inplace; refusing another installation or implicit JIT build')
torch.set_num_threads(settings.get('threads',8));torch.set_num_interop_threads(1);torch.manual_seed(1234)
from exllamav3 import Config,Model,Cache,Tokenizer,Generator,Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant

def serial(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)

status('config')
config=Config.from_directory(MODEL);model=Model.from_config(config)
draft=Model.from_config(config,component='mtp') if a.mode=='on' else None
# A draft window of two requires corresponding recurrent history for rewind.
history=max(draft.caps.get('default_draft_size',4) if draft else 0,settings.get('ndt',2) if draft else 0)
kv=settings.get('kv_bits',0)
cache_kw=dict(layer_type=CacheLayer_quant,k_bits=kv,v_bits=kv) if kv else {}
cache=Cache(model,max_num_tokens=16384,max_history=history,max_batch_size=1,**cache_kw)
status('load-model',history=history)
t=time.monotonic();model.load(device='cuda:0',progressbar=False,max_chunk_size=512);torch.cuda.synchronize()
prop=torch.cuda.get_device_properties(0)
assert prop.gcnArchName.split(':')[0]=='gfx1103',prop.gcnArchName
ablation={'attn':sum(getattr(m,'abl_attn',None) is not None for m in model.modules),
          'mlp':sum(getattr(m,'abl_mlp',None) is not None for m in model.modules)}
assert ablation=={'attn':61,'mlp':39},('Unexpected checkpoint ablation projections',ablation)
metadata={'torch':torch.__version__,'hip':torch.version.hip,'gpu':prop.name,'arch':prop.gcnArchName,
          'multiprocessor_count':prop.multi_processor_count,'model_load_s':time.monotonic()-t,
          'ablation_projections':ablation,'mode':a.mode,'context':16384,'cache_dtype':f'q{kv}' if kv else 'f16',
          'preset_file':a.preset_file,'preset':settings,
          'recurrent_history':history,'environment':{k:v for k,v in os.environ.items() if k.startswith(('EXL3_','OMP_','HIP_VISIBLE','ROCBLAS'))}}
draft_cache=None
if draft:
    draft_cache=Cache(draft,max_num_tokens=16384,max_batch_size=1,**cache_kw)
    status('load-mtp');draft.load(device='cuda:0',progressbar=False,max_chunk_size=512);torch.cuda.synchronize()
metadata['gpu_allocated_bytes']=torch.cuda.memory_allocated()
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
(OUT/'maps.txt').write_text(Path('/proc/self/maps').read_text())
tokenizer=Tokenizer.from_config(config)

def run(case,prompt,n):
    # Fresh page table and recurrent prefix cache for every request, outside the timed region.
    generator=Generator(model=model,cache=cache,tokenizer=tokenizer,max_batch_size=1,
       max_chunk_size=512,draft_model=draft,draft_cache=draft_cache,
       num_draft_tokens=settings.get('ndt',2) if draft else None,dynamic_draft_tokens=settings.get('dynamic',False),recurrent_cache_size=0)
    ids=tokenizer.encode(prompt,encode_special_tokens=True)
    job=Job(input_ids=ids,max_new_tokens=n,sampler=ComboSampler(temperature=0,top_p=1,top_k=1),
            seed=1234,stop_conditions=[])
    torch.cuda.synchronize();status(case,prompt_tokens=ids.shape[-1],tokens=n)
    first=None;last=None;pieces=[];tokens=[];started=time.monotonic();start_wall=time.time()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get('token_ids') is not None and r['token_ids'].numel():
                if first is None:first=time.monotonic()
                tokens.extend(r['token_ids'].reshape(-1).tolist())
            if r.get('text'):pieces.append(r['text'])
            if r.get('eos'):last=r
    torch.cuda.synchronize();ended=time.monotonic()
    assert last is not None,'No final event'
    produced=last['new_tokens'];assert produced==n,(produced,n)
    assert last.get('cached_tokens',0)==0,'Unexpected prompt-cache reuse'
    row={'mode':a.mode,'case':case,'prompt':prompt,'input_ids':ids.tolist(),'output_ids':tokens,
         'prompt_tokens':ids.shape[-1],'new_tokens':produced,'start_time':start_wall,'end_time':time.time(),
         'wall_s':ended-started,'first_token_s':None if first is None else first-started,
         'decode_s':None if first is None else ended-first,
         'decode_tokens_per_second':None if first is None else (produced-1)/(ended-first),
         'text':''.join(pieces),'last_event':serial(last)}
    (OUT/f'{case}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
    (OUT/f'{case}.txt').write_text(row['text'])
    status(case+'-done',new_tokens=produced,decode_tokens_per_second=row['decode_tokens_per_second'],
           accepted=last.get('accepted_draft_tokens'),rejected=last.get('rejected_draft_tokens'))
    generator.filter_pool.shutdown(wait=True)
    return row

small='<|im_start|>user\nExplique en une phrase le rôle de la mémoire vive.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
if a.profile_smoke:
    status('profile-smoke')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        run('profile',small,4)
    prof.export_chrome_trace(str(OUT/'gpu-trace.json'))
    gpu_events=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
    evidence={'gpu_event_count':len(gpu_events),'gpu_event_examples':[{'name':e.name,'device_index':e.device_index,
       'duration_us':e.time_range.elapsed_us()} for e in gpu_events[:20]]}
    (OUT/'gpu-profile-summary.json').write_text(json.dumps(evidence,indent=2))
    status('profile-complete',gpu_event_count=len(gpu_events))
else:
    run('warmup',small,16 if a.smoke else 24)
if not a.smoke:
    prompts={'linux': '<|im_start|>user\nExplique en français à un débutant comment Linux utilise la mémoire vive : mémoire des programmes, cache disque, mémoire disponible et swap. Donne des exemples concrets et explique pourquoi une RAM presque pleine ne signifie pas forcément un problème. Termine par les précautions utiles avant de fermer un programme. Développe ta réponse en plusieurs paragraphes.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'python': '<|im_start|>user\nÉcris un script Python 3 complet qui lit un fichier CSV contenant les colonnes date, categorie et montant, valide les lignes et affiche le total par catégorie. Utilise uniquement la bibliothèque standard, Decimal pour les montants et argparse pour le chemin du fichier. Gère les fichiers absents et les lignes invalides. Fournis uniquement le code, avec des commentaires en français.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'}
    for case,prompt in prompts.items():
        run(case,prompt,256)
status('complete')
