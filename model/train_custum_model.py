import torch
from pathlib import Path
import argparse
import sys
import glob
import os

print("="*70)
print("🔧 ENREGISTRAMENTO DOS MÓDULOS")
print("="*70 + "\n")

try:
    import mobilenetv3
    print("✅ Module mobilenetv3 importado\n")
    
    available = [x for x in dir(mobilenetv3) if not x.startswith('_')]
    print(f"Classes encontradas: {available}\n")
    
    # Guarda as classes
    modules_dict = {}
    for attr_name in available:
        attr = getattr(mobilenetv3, attr_name)
        modules_dict[attr_name] = attr
        if isinstance(attr, type):
            print(f"✅ {attr_name}")
    
    print()
    
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)

# ⭐ PATCH CRÍTICO: Modifica parse_model antes de ultralytics
print("="*70)
print("🔨 PATCH DO PARSE_MODEL")
print("="*70 + "\n")

try:
    from ultralytics.nn import tasks
    
    original_parse_model = tasks.parse_model
    
    def patched_parse_model(d, ch, verbose=True):
        """parse_model com suporte a módulos customizados"""
        import torch
        import torch.nn as nn
        from copy import deepcopy
        
        # Mescla os módulos customizados com os padrões
        custom_modules = modules_dict.copy()
        custom_modules.update({
            'nn': nn,
            'torch': torch,
            'F': torch.nn.functional,
        })
        
        # Chama o original mas injeta nossos módulos no globals
        old_globals = {}
        for key, val in custom_modules.items():
            if key not in tasks.__dict__:
                old_globals[key] = tasks.__dict__.get(key)
                tasks.__dict__[key] = val
        
        try:
            return original_parse_model(d, ch, verbose=verbose)
        finally:
            # Restaura
            for key in custom_modules:
                if key in old_globals and old_globals[key] is None:
                    del tasks.__dict__[key]
                elif key in old_globals:
                    tasks.__dict__[key] = old_globals[key]
    
    tasks.parse_model = patched_parse_model
    print("✅ parse_model patched\n")
    
except Exception as e:
    print(f"⚠️ Aviso ao fazer patch: {e}\n")

print("="*70)
print("📦 IMPORTAÇÃO DO YOLO")
print("="*70 + "\n")

try:
    from ultralytics import YOLO
    print("✅ YOLO importado\n")
except ImportError as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

print("="*70)
print("📁 VERIFICAÇÃO DOS ARQUIVOS")
print("="*70 + "\n")

current_dir = Path(__file__).parent
yaml_path = current_dir / 'mbnet.yaml'
data_yaml = current_dir / 'custom_data.yaml'

if not yaml_path.exists():
    print(f"❌ {yaml_path} não existe!")
    sys.exit(1)
print(f"✅ mbnet.yaml")

if not data_yaml.exists():
    print(f"❌ {data_yaml} não existe!")
    sys.exit(1)
print(f"✅ custom_data.yaml\n")

print("="*70)
print("🖥️ GPU/CPU")
print("="*70 + "\n")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("⚠️ Sem GPU - CPU será usado")
    device = 'cpu'

print()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--resume', action='store_true', help='Reprendre l\'entraînement depuis le dernier checkpoint')
args = parser.parse_args()

print("="*70)
print("⏳ CARREGAMENTO DO MODELO")
print("="*70 + "\n")

# Recherche automatique du dernier checkpoint valide
def find_latest_checkpoint():
    checkpoint_list = glob.glob('runs/detect/yolov8_mobilenetv3*/weights/last.pt')
    if not checkpoint_list:
        return None
    
    # Trier par date de modification pour avoir le plus récent
    latest_ckpt = max(checkpoint_list, key=os.path.getmtime)
    return latest_ckpt

checkpoint_path = find_latest_checkpoint()

try:
    if args.resume and checkpoint_path:
        print(f"🔄 Tentativa de retomada do checkpoint: {checkpoint_path}")
        # Vérification rapide du nc pour éviter les erreurs de mismatch
        try:
            ckpt_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            ckpt_nc = ckpt_data.get('ema', ckpt_data.get('model')).nc
            print(f"📊 Checkpoint détecté avec nc={ckpt_nc}")
            if ckpt_nc != 5: # On sait que le dataset actuel a nc=5
                 print(f"⚠️ ATTENTION: Le checkpoint {checkpoint_path} a nc={ckpt_nc} mais custom_data.yaml a nc=5.")
                 print(f"⚠️ La reprise risque d'échouer. Recherche d'un autre checkpoint...")
                 # On pourrait filtrer ici pour trouver un checkpoint avec nc=5
                 valid_checkpoints = []
                 for cp in glob.glob('runs/detect/yolov8_mobilenetv3*/weights/last.pt'):
                     try:
                         d = torch.load(cp, map_location='cpu', weights_only=False)
                         if (d.get('ema') or d.get('model')).nc == 5:
                             valid_checkpoints.append(cp)
                     except: continue
                 
                 if valid_checkpoints:
                     checkpoint_path = max(valid_checkpoints, key=os.path.getmtime)
                     print(f"✅ Nouveau checkpoint trouvé: {checkpoint_path}")
                 else:
                     print(f"❌ Aucun checkpoint avec nc=5 trouvé. Reprise impossible.")
                     checkpoint_path = None
        except Exception as e:
            print(f"⚠️ Erreur lors de la vérification du checkpoint: {e}")

    if args.resume and checkpoint_path:
        model = YOLO(str(checkpoint_path))
        is_resume = True
    else:
        if args.resume:
            print(f"⚠️ Checkpoint valide non trouvé. Iniciando do zero.")
        print(f"🆕 Iniciando novo treinamento com {yaml_path}")
        model = YOLO(str(yaml_path))
        is_resume = False
    print("✅ Modelo carregado com sucesso\n")
except Exception as e:
    print(f"❌ ERRO ao carregar modelo: {e}")
    sys.exit(1)

print("="*70)
print("🚀 TREINAMENTO")
print("="*70 + "\n")

print(f"Epochs: {args.epochs}")
print(f"Batch: {args.batch}")
print(f"Tamanho da imagem: {args.imgsz}")
print(f"Device: {device}")
print(f"Resume: {is_resume}\n")

try:
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=20,
        save=True,
        verbose=True,
        name='yolov8_mobilenetv3',
        project='runs/detect',
        resume=is_resume,
        exist_ok=True # Permet de continuer dans le même dossier si possible
    )
    
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
