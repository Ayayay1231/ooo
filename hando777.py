import cv2
import mediapipe as mp
from pyfirmata import Arduino, util
import time

port = "COM4"
LED_PINS = [5,6,9,10,11]


try:
    board = Arduino(PORT)
    print(f"Connected to Arduino on port {PORT}. Initializing {len(LED_PINS)} LEDs.")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    board = None

# --- 2. การตั้งค่า MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 3. การกำหนด Landmark สำหรับแต่ละนิ้ว ---
# ตำแหน่ง Tip (ยอดนิ้ว) และ PIP (ข้อนิ้วที่สอง)
FINGER_LANDMARKS = {
    # Tip (ยอด) : PIP (ข้อ 2) : MCP (ข้อ 1 - สำหรับนิ้วโป้ง)
    "Thumb": (4, 3, 2),    
    "Index": (8, 6),
    "Middle": (12, 10),
    "Ring": (16, 14),
    "Pinky": (20, 18)
}

# --- ฟังก์ชันช่วย: ตรวจสอบว่านิ้วยกขึ้นหรือไม่ ---
def is_finger_raised(hand_landmarks, finger_name):
    """
    ตรวจสอบนิ้ว 5 นิ้ว โดยใช้วิธีการต่างกันสำหรับนิ้วโป้งและนิ้วอื่นๆ
    """
    landmarks = hand_landmarks.landmark
    
    # สำหรับนิ้ว Index, Middle, Ring, Pinky (ใช้วิธีเปรียบเทียบ Y-coordinate)
    if finger_name in ["Index", "Middle", "Ring", "Pinky"]:
        tip_idx, pip_idx = FINGER_LANDMARKS[finger_name]
        # Y-coordinate น้อยกว่าแปลว่าอยู่สูงกว่า
        return landmarks[tip_idx].y < landmarks[pip_idx].y
    
    # สำหรับนิ้ว Thumb (นิ้วโป้ง)
    elif finger_name == "Thumb":
        tip_idx, _, mcp_idx = FINGER_LANDMARKS[finger_name]
        
        # ตรวจสอบว่ามือหงาย (Palm up) หรือคว่ำ (Palm down) เพื่อเลือกการเปรียบเทียบ X หรือ Y
        # วิธีที่ง่าย: เทียบ X-coordinate ของ Tip (4) กับ MCP (2) 
        # ถ้า Tip อยู่ด้านซ้าย (Xs น้อยกว่า) MCP และมือมีการวางตัวที่เหมาะสม
        # ให้ใช้การเปรียบเทียบ Y เหมือนนิ้วอื่นเป็นหลักในการนับแบบง่าย
        # แต่เพื่อความง่ายในการนับ 1-5 แบบตรงๆ เราใช้การเทียบ Y เหมือนนิ้วอื่น
        return landmarks[tip_idx].y < landmarks[mcp_idx].y


# --- 4. การตั้งค่ากล้องและการทำงานหลัก ---
cap = cv2.VideoCapture(0)
print("Start gesture recognition. Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_finger_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # ตรวจสอบนิ้วทั้ง 5
            for finger_name in FINGER_LANDMARKS.keys():
                if is_finger_raised(hand_landmarks, finger_name):
                    current_finger_count += 1
            
            # วาดจุด Landmark บนภาพ
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- 5. สั่งงาน Arduino ควบคุม LED 5 ดวง ---
            if board:
                # เปิด LED ตามจำนวนนิ้วที่ยกขึ้น
                for i in range(5):
                    if i < current_finger_count:
                        # เปิดไฟ (1) สำหรับนิ้วที่ยกขึ้น
                        board.digital[LED_PINS[i]].write(1)
                    else:
                        # ปิดไฟ (0) สำหรับนิ้วที่เหลือ
                        board.digital[LED_PINS[i]].write(0)

            # แสดงจำนวนนิ้วที่นับได้บนหน้าจอ
            cv2.putText(image, f"Fingers: {current_finger_count}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # ใช้แถบสถานะเพื่อแสดงการควบคุม
            status_text = f"LEDs ON: {current_finger_count} / {len(LED_PINS)}"
            cv2.putText(image, status_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('MediaPipe 5-Finger Detection', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 6. การปิดระบบ ---
cap.release()
cv2.destroyAllWindows()
if board:
    # ปิด LED ทั้งหมดก่อนออกจากโปรแกรม
    for pin in LED_PINS:
        board.digital[pin].write(0) 
    board.exit()
    print("Arduino connection closed. All LEDs turned off.")
