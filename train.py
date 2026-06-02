import argparse
# Wir importieren die fertige Funktion aus deiner utils/dataloader.py
from utils.dataloader import create_data_loader 

def main():
    # 1. Deine Konfiguration (Args) festlegen
    args = argparse.Namespace(
        train_wav_txt="/app/data/dev_train_clean.txt",
        feature_config="/app/configs/feature.json",
        batch_size=4,             # 4 Samples werden zu einem Batch zusammengefasst
        train_wav_length=1.27,
        fft_size=512,
        stft_hop_size=240,
        feature="amp_phasediff",
        class_num=13,
    )

    # 2. DataLoader erstellen (ruft intern SELDDataSet(args) auf)
    dataloader = create_data_loader(args)
    
    # 3. So benutzt du ihn in deiner Trainingsschleife (z.B. für 100 Epochen):
    epochs = 100
    for epoch in range(epochs):
        print(f"\n🔄 Starte Epoche {epoch + 1}/{epochs}")
        
        # Der DataLoader mischt die Daten in jeder Epoche neu und liefert uns fertige Batches
        for batch_idx, batch in enumerate(dataloader):
            
            # Wir entpacken die 4 Dinge, die __getitem__ zurückgibt:
            input_spec, frame_out_float, label_float, names = batch
            
            # --- AB HIER GEHT DEIN TRAINING LOS ---
            # 1. Daten an die Grafikkarte (GPU) senden:
            # audio = input_spec.cuda()
            # video = frame_out_float.cuda()
            # targets = label_float.cuda()
            
            # 2. Die Daten in dein Fusions-Modell füttern (Early, Mid oder Late):
            # outputs = mein_fusions_modell(audio, video)
            
            # 3. Loss berechnen und Gewichte anpassen...
            
            # Kleiner Print, um den Fortschritt zu sehen:
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Audio-Shape im Batch = {input_spec.shape}")
        
        # Nach einem kompletten Durchlauf (Dataloader ist leer) endet die Epoche



if __name__ == "__main__":
    main()
