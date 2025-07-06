
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

video_path = None
cap = None
is_playing = False
current_frame = 0


model = YOLO("yolov8n.pt")

ctk.set_appearance_mode("dark")           
ctk.set_default_color_theme("blue")    


root = ctk.CTk()
root.title("Détection des Objets/Visages - YOLOv8")
root.geometry("900x750")


def parcourir_video():
    global video_path, cap, current_frame
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        current_frame = 0 
        jouer_video()


def jouer_video():
    global is_playing, current_frame
    if cap is not None:
        is_playing = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        lire_frame()

def pause_video():
    global is_playing
    is_playing = False

def lire_frame():
    global is_playing, current_frame
    if is_playing:
        ret, frame = cap.read()
        if ret:
            current_frame += 1
            frame = detecter_objets(frame)
            afficher_frame(frame)
            root.after(25, lire_frame)
        else:
            arreter_video()


def arreter_video():
    global is_playing, current_frame
    is_playing = False
    current_frame = 0  
    if cap is not None:
        cap.release()



def detecter_objets(frame):
    results = model(frame)
    objets_detectes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(class_id)]
            objets_detectes.append(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    

    resultat_text.set(f"Objets détectés : {', '.join(objets_detectes)}\nNombre total : {len(objets_detectes)}")
    return frame


def afficher_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.thumbnail((500, 500))  
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk




main_frame = ctk.CTkFrame(root)
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

bouton_parcourir = ctk.CTkButton(main_frame, text="📂 Parcourir une vidéo",
                                 command=parcourir_video, fg_color="#1E90FF", hover_color="#104E8B",
                                 font=("Arial", 14, "bold"), height=40, corner_radius=10)
bouton_parcourir.pack(pady=10)

panel = ctk.CTkLabel(main_frame, text="", width=500, height=400, fg_color="#2C2F33", corner_radius=10)
panel.pack(pady=10)

button_frame = ctk.CTkFrame(main_frame)
button_frame.pack(pady=10)

bouton_jouer = ctk.CTkButton(button_frame, text="▶️ Play",
                             command=jouer_video, fg_color="green", hover_color="darkgreen",
                             font=("Arial", 14, "bold"), width=100, height=40, corner_radius=10)
bouton_jouer.pack(side="left", padx=20)

bouton_pause = ctk.CTkButton(button_frame, text="⏸ Pause",
                             command=pause_video, fg_color="red", hover_color="darkred",
                             font=("Arial", 14, "bold"), width=100, height=40, corner_radius=10)
bouton_pause.pack(side="right", padx=20)

resultat_text = ctk.StringVar()
resultat_label = ctk.CTkLabel(main_frame, textvariable=resultat_text, fg_color="black",
                              font=("Arial", 14, "bold"), width=400, height=50, corner_radius=10)
resultat_label.pack(pady=10)


root.mainloop()







