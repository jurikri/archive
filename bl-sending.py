# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:02:38 2023

@author: PC
"""
#%%

import serial # pip install pyserial
import time

def ms_blsend(data, port, baudrate):
    ser = serial.Serial(port, baudrate)
    try:
        ser.write(data.encode('ascii'))
        print("Data sent:", data)

        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting).decode('ascii')
            print("Received:", response)
    
    except Exception as e:
        print("Error:", e)
    
    finally:
        # 연결 종료
        ser.close()

def find_portnum():
    # 시리얼 포트 설정
    for comnum in range(0, 13):
        print(comnum)
        port = "COM" + str(comnum)  # 포트 번호
        baudrate = 9600  # 보드레이트

        try:
            _ = serial.Serial(port, baudrate)
            # ms_blsend("CS229E", ser)
            return comnum
        except:
            pass
    return None

comnum = find_portnum()

print('comnum ->', comnum)
port = "COM" + str(comnum)  # 포트 번호
baudrate = 9600  # 보드레이트
ms_blsend("CT110E", port, baudrate)

if True: # 5초간 VNS 자극 후, 종료
    print('on')
    ms_blsend("CS226E", port, baudrate)
    time.sleep(600)
    
    print('off')
    ms_blsend("CT110E", port, baudrate)
    time.sleep(0.1)
    
