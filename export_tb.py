import os
import csv
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Pfad zum Ordner, in dem TensorBoard seine Logs ablegt
LOG_DIR = "/app/results/"
OUTPUT_DIR = "/app/results/val/"

def main():
    # 1. Suche nach allen tfevents-Logdateien
    event_files = glob.glob(os.path.join(LOG_DIR, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"Keine TensorBoard-Dateien in {LOG_DIR} gefunden!")
        return

    # Nimm die neueste Logdatei deines letzten Trainings
    latest_event = max(event_files, key=os.path.getmtime)
    print(f"Lese TensorBoard-Log: {latest_event}")

    ea = EventAccumulator(latest_event)
    ea.Reload()

    # 2. Alle verfügbaren Skalar-Metriken auslesen
    tags = ea.Tags().get("scalars", [])
    print(f"Gefundene Metriken: {tags}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Jede gefundene Loss-Metrik als CSV exportieren
    for tag in tags:
        # Filter: Wir wollen die Loss-Werte exportieren
        if "loss" in tag.lower() or "train" in tag.lower():
            events = ea.Scalars(tag)
            safe_tag_name = tag.replace("/", "_").replace(" ", "_")
            out_file = os.path.join(OUTPUT_DIR, f"tensorboard_{safe_tag_name}.csv")
            
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Value"])
                for event in events:
                    writer.writerow([event.step, event.value])
            
            print(f"Erfolgreich als CSV gespeichert: {out_file}")

if __name__ == "__main__":
    main()