import cv2
import numpy as np
from gpiozero import LED
from picamera2 import Picamera2
import time
from rplidar import RPLidar  # Для работы с лидаром

# Настройки GPIO
RELAY_SLOW_PIN = 17  # Реле для снижения скорости
RELAY_STOP_PIN = 18  # Реле для экстренного торможения
LED_WARNING_PIN = 22  # Светодиод предупреждения

# Пороговые значения TTC (сек)
WARNING_TTC = 10.0  # Время для предупреждения
STOP_TTC = 1.0      # Время для экстренной остановки

# Настройки камеры
FOCAL_LENGTH = 1000      # Фокусное расстояние (калибровать!)
KNOWN_HEAD_WIDTH = 15.0  # Средняя ширина головы в см
CASCADE_PATH = '/home/pi/haarcascade_frontalface_default.xml'

class SafetySystem:
    def __init__(self):
        # Инициализация GPIO
        self.slow_relay = LED(RELAY_SLOW_PIN)
        self.stop_relay = LED(RELAY_STOP_PIN)
        self.warning_led = LED(LED_WARNING_PIN)
        
        # Инициализация камеры
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(config)
        
        # Инициализация детектора лиц
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self.face_cascade.empty():
            raise Exception("Не удалось загрузить каскадный классификатор!")
        
        # Инициализация лидара
        self.lidar = RPLidar('/dev/ttyUSB0')
        self.current_lidar_distance = None
        self.prev_lidar_distance = None
        self.last_measurement_time = time.time()
        
        # Переменные для расчета TTC
        self.prev_distance = None
        self.prev_time = None
        self.ttc = None
        
    def get_lidar_distance(self):
        """Получение расстояния с лидара"""
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=100):
                for _, angle, distance in scan:
                    if 160 < angle < 200:  # Область прямо перед роботом
                        self.prev_lidar_distance = self.current_lidar_distance
                        self.current_lidar_distance = distance / 10  # Переводим в см
                        self.last_measurement_time = time.time()
                        return self.current_lidar_distance
        except Exception as e:
            print(f"Ошибка лидара: {e}")
            return None
    
    def calculate_distance(self, face_width):
        """Вычисление расстояния по размеру лица"""
        return (KNOWN_HEAD_WIDTH * FOCAL_LENGTH) / face_width
    
    def calculate_ttc(self, current_distance):
        """Вычисление времени до столкновения (TTC)"""
        now = time.time()
        if self.prev_distance is not None and self.prev_time is not None:
            time_diff = now - self.prev_time
            distance_diff = self.prev_distance - current_distance
            
            if time_diff > 0 and distance_diff > 0:
                speed = distance_diff / time_diff  # Скорость сближения (см/сек)
                if speed > 0:
                    self.ttc = current_distance / speed
                    return self.ttc
        
        self.prev_distance = current_distance
        self.prev_time = now
        return None
    
    def control_system(self, ttc):
        """Управление реле на основе TTC"""
        # Все реле выключены по умолчанию
        self.stop_relay.off()
        self.slow_relay.off()
        self.warning_led.off()
        
        if ttc is None:
            return "NO DATA"
        
        if ttc <= STOP_TTC:
            self.stop_relay.on()
            self.warning_led.on()
            return "EMERGENCY STOP!"
        elif ttc <= WARNING_TTC:
            self.slow_relay.on()
            self.warning_led.blink(on_time=0.5, off_time=0.5)
            return "WARNING: SLOW DOWN"
        else:
            return "SAFE"
    
    def run(self):
        try:
            self.picam2.start()
            
            while True:
                # Получаем данные с лидара в отдельном потоке
                lidar_distance = self.get_lidar_distance()
                
                # Получаем кадр с камеры
                frame = self.picam2.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Детекция лиц
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Вычисление расстояния
                        camera_distance = self.calculate_distance(w)
                        
                        # Используем данные лидара если они доступны, иначе камеру
                        if lidar_distance is not None:
                            current_distance = lidar_distance
                        else:
                            current_distance = camera_distance
                        
                        # Расчет TTC
                        ttc = self.calculate_ttc(current_distance)
                        status = self.control_system(ttc)
                        
                        # Отрисовка информации
                        color = (0, 255, 0)  # Зеленый по умолчанию
                        if status == "WARNING: SLOW DOWN":
                            color = (0, 255, 255)  # Желтый
                        elif status == "EMERGENCY STOP!":
                            color = (0, 0, 255)  # Красный
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        info_text = f"Dist: {current_distance:.1f}cm, TTC: {ttc:.1f}s" if ttc else f"Dist: {current_distance:.1f}cm"
                        cv2.putText(frame, f"{info_text} - {status}", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, color, 2)
                else:
                    status = self.control_system(None)
                    cv2.putText(frame, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (255, 255, 255), 2)
                
                # Отображение кадра
                cv2.imshow('Safety System', frame)
                
                # Выход по нажатию 'q'
                if cv2.waitKey(1) == ord('q'):
                    break
                
        finally:
            # Корректное завершение
            self.stop_relay.off()
            self.slow_relay.off()
            self.warning_led.off()
            self.picam2.stop()
            self.lidar.stop()
            self.lidar.disconnect()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SafetySystem()
    system.run()