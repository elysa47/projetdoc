import os
from pathlib import Path
import yaml

def extract_class_ids(labels_dir):
    """Extrait tous les IDs de classe présents dans les fichiers .txt"""
    print(f"🔍 Analyse du dossier: {labels_dir}\n")
    
    class_ids = set()
    txt_files = list(Path(labels_dir).glob('*.txt'))
    
    print(f"📄 Nombre de fichiers .txt trouvés: {len(txt_files)}")
    
    if len(txt_files) == 0:
        print("❌ Aucun fichier .txt trouvé !")
        print(f"   Vérifiez que le chemin existe: {labels_dir}")
        return None
    
    # Barre de progression
    for i, txt_file in enumerate(txt_files, 1):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        # Format YOLO: class_id x y w h
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except Exception as e:
            print(f"   ⚠️ Erreur dans {txt_file.name}: {e}")
        
        # Affiche la progression
        if i % 50 == 0:
            print(f"   ✓ {i} fichiers traités...")
    
    print(f"   ✓ {len(txt_files)} fichiers traités\n")
    
    return sorted(list(class_ids))

def create_yaml_file(dataset_path, class_ids):
    """Crée le fichier custom_data.yaml"""
    
    print("=" * 70)
    print("📝 CRÉATION DU FICHIER custom_data.yaml")
    print("=" * 70)
    
    print(f"\n✅ Classes détectées (IDs): {class_ids}")
    print(f"   Nombre total: {len(class_ids)} classe(s)\n")
    
    # Demande les noms des classes
    class_names = {}
    
    print("❓ Entrez le nom pour chaque classe:")
    print("   (ou appuyez sur Entrée pour utiliser le nom par défaut)\n")
    print("   Exemples: Healthy, Early_Blight, Late_Blight, etc.\n")
    
    for class_id in class_ids:
        default_name = f"class_{class_id}"
        
        try:
            user_input = input(f"   ID {class_id} -> ").strip()
            name = user_input if user_input else default_name
        except:
            name = default_name
        
        class_names[class_id] = name
        print(f"            ✅ {class_id} = {name}")
    
    # Configuration YAML
    data_config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Sauvegarde
    output_file = 'custom_data.yaml'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, 
                     default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Fichier créé avec succès: {output_file}\n")
        
        # Affiche le contenu
        print("=" * 70)
        print("📄 CONTENU DU FICHIER YAML")
        print("=" * 70)
        print()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde: {e}")
        return False

def verify_dataset(dataset_base):
    """Vérifie la structure du dataset"""
    print("\n" + "=" * 70)
    print("📂 VÉRIFICATION DE LA STRUCTURE DU DATASET")
    print("=" * 70 + "\n")
    
    paths_to_check = {
        'images/train': os.path.join(dataset_base, 'images', 'train'),
        'images/val': os.path.join(dataset_base, 'images', 'val'),
        'labels/train': os.path.join(dataset_base, 'labels', 'train'),
        'labels/val': os.path.join(dataset_base, 'labels', 'val'),
    }
    
    for path_name, path_full in paths_to_check.items():
        if os.path.exists(path_full):
            # Compte les fichiers
            files = list(Path(path_full).glob('*'))
            print(f"✅ {path_name:<20} ({len(files)} fichiers)")
        else:
            print(f"❌ {path_name:<20} (manquant)")
    
    print()

def main():
    print("\n" + "=" * 70)
    print("🎯 EXTRACTEUR DE CLASSES - YOLO FORMAT")
    print("=" * 70 + "\n")
    
    # Votre chemin exact
    dataset_base = r'D:\YOLOV8MBNETV3\dataset'
    labels_train = os.path.join(dataset_base, 'labels', 'train')
    
    # Vérifie que le dossier existe
    if not os.path.exists(labels_train):
        print(f"❌ ERREUR: Le dossier n'existe pas!")
        print(f"   Chemin attendu: {labels_train}\n")
        print("💡 Solutions:")
        print("   1. Vérifiez que le chemin est correct")
        print("   2. Assurez-vous que les fichiers .txt existent")
        print("   3. Vérifiez la structure du dataset\n")
        return
    
    print(f"✅ Dossier trouvé: {labels_train}\n")
    
    # Vérifie la structure complète
    verify_dataset(dataset_base)
    
    # Extrait les classes
    print("=" * 70)
    print("🔍 EXTRACTION DES CLASSES")
    print("=" * 70 + "\n")
    
    class_ids = extract_class_ids(labels_train)
    
    if class_ids is None:
        print("\n❌ Impossible d'extraire les classes")
        return
    
    # Crée le fichier YAML
    if create_yaml_file(dataset_base, class_ids):
        print("\n🎉 PRÊT POUR L'ENTRAÎNEMENT!")
        print("\nCommande suivante:")
        print("   python train_custom_model.py --data custom_data.yaml --cfg mbnet.yaml")

if __name__ == "__main__":
    main()
