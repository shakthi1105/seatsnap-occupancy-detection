from ultralytics import YOLO
import cv2
import csv
import os

PERSON = 0
CHAIR = 1
TABLE = 2
PARTITION = 3

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

def assign_seat_ids(chairs, row_threshold=80):
    centers = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in chairs]
    rows = []
    for i, (cx, cy) in enumerate(centers):
        placed = False
        for row in rows:
            if abs(cy - row["y_mean"]) < row_threshold:
                row["indices"].append(i)
                row["y_vals"].append(cy)
                row["y_mean"] = sum(row["y_vals"]) / len(row["y_vals"])
                placed = True
                break
        if not placed:
            rows.append({"indices":[i], "y_vals":[cy], "y_mean":cy})
    rows = sorted(rows, key=lambda r: r["y_mean"])
    seat_order = []
    seat_id = 1
    for row in rows:
        ids_sorted = sorted(row["indices"], key=lambda idx: centers[idx][0])
        for idx in ids_sorted:
            seat_order.append((seat_id, chairs[idx]))
            seat_id += 1
    return seat_order

def compute_seat_status(chairs_with_ids, persons):
    out = []
    for seat_id, box in chairs_with_ids:
        occupied = any(iou(box, p) > 0.05 for p in persons)
        out.append({"id": seat_id, "bbox": box, "occupied": occupied})
    return out

def generate_seat_level_csv(seat_status, image_name, output_file="seat_level_occupancy.csv"):
    """Writes the seat-by-seat occupancy into CSV."""
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_name", "seat_id", "occupied"])
        for s in seat_status:
            writer.writerow([
                image_name,
                s["id"],
                "yes" if s["occupied"] else "no"
            ])
    print(f"\nCSV updated: {output_file}")

def train_model():
    model = YOLO("yolov8m.pt")
    model.train(
        data="library.yaml",
        imgsz=1280,
        epochs=80,
        batch=4,
        workers=4,
        optimizer="AdamW",
        patience=20,
        project="runs/library_train",
        name="seat_model"
    )
    print("\nTraining complete!")

def run_inference(image_path):
    model_path = "runs/library_train/seat_model/weights/best.pt"
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    results = model(img, conf=0.4)[0]
    chairs, persons, tables, partitions = [], [], [], []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = int(cls)
        if cls == CHAIR:
            chairs.append([x1,y1,x2,y2])
        elif cls == PERSON:
            persons.append([x1,y1,x2,y2])
        elif cls == TABLE:
            tables.append([x1,y1,x2,y2])
        elif cls == PARTITION:
            partitions.append([x1,y1,x2,y2])
    chairs_with_ids = assign_seat_ids(chairs)
    seat_status = compute_seat_status(chairs_with_ids, persons)
    print("\n=== SEAT STATUS ===")
    for s in seat_status:
        print(f"Seat {s['id']}: {'OCCUPIED' if s['occupied'] else 'FREE'}")
    generate_seat_level_csv(seat_status, os.path.basename(image_path))
    cv2.imwrite("output.jpg", img)
    print("\nSaved annotated output as output.jpg")
    return seat_status

if __name__ == "__main__":
    print("1 = Train Model")
    print("2 = Run Inference + CSV Output")
    choice = input("Select option: ")
    if choice == "1":
        train_model()
    elif choice == "2":
        img = input("Enter image path: ")
        run_inference(img)
    else:
        print("Invalid choice")
