# AutoTormoZ
Automatic braking system for electric scooter
### **Описание проекта для GitHub**  

### **Русская версия**  

#### **Система предотвращения столкновений на Raspberry Pi 4**  
Этот проект реализует алгоритм определения времени до столкновения (Time To Collision, **TTC**) с помощью камеры и лидара. Система анализирует скорость сближения с объектом (например, человеком) и автоматически активирует торможение при критическом сближении.  

#### **📌 Основные функции**  
- **Детекция лиц** с помощью каскадов Haar (OpenCV)  
- **Определение расстояния** двумя методами:  
  - По размеру головы на камере (геометрический расчет)  
  - С помощью лидара (точные измерения)  
- **Расчет TTC** (времени до столкновения) на основе изменения расстояния  
- **Трехуровневая система безопасности**:  
  - **Предупреждение** (TTC ≤ 10 сек) – активация реле снижения скорости  
  - **Экстренное торможение** (TTC ≤ 1 сек) – полная остановка  
- **Визуализация** (OpenCV):  
  - Цветные рамки (зеленый/желтый/красный) в зависимости от опасности  
  - Вывод расстояния и TTC на экран  

#### **🛠 Технологии**  
- **Raspberry Pi 4** (с камерой)  
- **Лидар RPLIDAR** (для точного измерения расстояния)  
- **Python 3 + OpenCV** (обработка изображений)  
- **GPIO Zero** (управление реле и светодиодами)  

#### **🚀 Установка и запуск**  
```bash
# Установка зависимостей  
pip install opencv-python gpiozero rplidar-roboticia  

# Запуск системы  
python safety_system.py  
```  

---

### **English Version**  

#### **Collision Avoidance System for Raspberry Pi 4**  
This project implements a **Time To Collision (TTC)** detection algorithm using a camera and LiDAR. The system analyzes the closing speed with an object (e.g., a person) and automatically triggers braking when a critical distance is reached.  

#### **📌 Key Features**  
- **Face detection** using Haar cascades (OpenCV)  
- **Distance measurement** via two methods:  
  - Head size estimation (geometric calculation)  
  - LiDAR (precise distance data)  
- **TTC calculation** based on relative speed  
- **Three-stage safety system**:  
  - **Warning** (TTC ≤ 10 sec) – activates speed reduction relay  
  - **Emergency stop** (TTC ≤ 1 sec) – full braking  
- **Visualization** (OpenCV):  
  - Colored bounding boxes (green/yellow/red) based on risk  
  - Real-time distance and TTC display  

#### **🛠 Technologies**  
- **Raspberry Pi 4** (with camera module)  
- **RPLIDAR** (for accurate distance sensing)  
- **Python 3 + OpenCV** (image processing)  
- **GPIO Zero** (relay & LED control)  

#### **🚀 Setup & Run**  
```bash
# Install dependencies  
pip install opencv-python gpiozero rplidar-roboticia  

# Run the system  
python safety_system.py  
```  

---

### **📂 Project Structure**  
```
safety_system/  
├── safety_system.py    # Main script  
├── haarcascade_frontalface_default.xml  # Haar cascade model  
├── README.md           # Documentation  
└── requirements.txt    # Dependencies  
```  

### **📜 License**  
**MIT** – Open-source for educational and commercial use.  

---
Would you like to add any badges (e.g., for CI, PyPI) or extend the hardware setup section? 🚀
