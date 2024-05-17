import bluetooth
import serial
import serial.tools.list_ports
from pyOpenBCI import OpenBCICyton

# 블루투스 기기 검색
def find_bluetooth_device(target_name):
    print("Searching for devices...")
    nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=False)
    
    for addr, name in nearby_devices:
        print(f"Found {name} - {addr}")
        if target_name == name:
            print(f"Target device '{target_name}' found. Address: {addr}")
            return addr
    
    print(f"Target device '{target_name}' not found.")
    return None

# 사용 가능한 직렬 포트 검색
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

# 특정 직렬 포트에 데이터 전송
def send_data_via_serial(port_name, data, baud_rate=9600):
    try:
        with serial.Serial(port_name, baud_rate) as ser:
            print(f"Connected to {port_name}")
            ser.write(data.encode())
            print(f"Data sent: {data}")
    except serial.SerialException as e:
        print(f"Failed to send data to {port_name}. Error: {e}")

#%%
def main():
    target_name = "BB24_200"  # 목표 블루투스 기기 이름
    target_addr = find_bluetooth_device(target_name)
    
    if target_addr:
        print(f"Target Bluetooth device '{target_name}' found with address {target_addr}")
        
        # 사용 가능한 모든 직렬 포트 나열
        ports = list_serial_ports()
        print("Available serial ports:", ports)
        
        for port in ports:
            print(f"Trying port {port}...")
            try:
                OpenBCICyton(port=port, daisy=False)
            except serial.SerialException:
                print('pass')

if __name__ == "__main__":
    main()
    
    
#%%

import threading
import time
from pyOpenBCI import OpenBCICyton

class TimeoutException(Exception):
    pass

def connect_with_timeout(board, timeout):
    def target():
        try:
            board.start_stream()  # Replace with the actual connection method if different
        except Exception as e:
            board.error = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException("Connection attempt timed out")
    if hasattr(board, 'error'):
        raise board.error

port = 'COM3'  # Replace with your actual port
daisy = False

try:
    board = OpenBCICyton(port=port, daisy=daisy)
    connect_with_timeout(board, timeout=10)  # 10 seconds timeout
    print("Successfully connected to OpenBCI Cyton board.")
except TimeoutException as e:
    print(f"Failed to connect: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

































