# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:16:10 2024

@author: PC
"""

import bluetooth # https://github.com/pybluez/pybluez
import serial
import serial.tools.list_ports
from pyOpenBCI import OpenBCICyton
# import subprocess
import time
# from win32com.client import Dispatch
import threading

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

def pair_bluetooth_device(addr):
    import bluetooth
    # 블루투스 소켓 생성
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    
    # 서버 소켓 연결 시도
    port = 1
    sock.connect((addr, port))
    print(f"Connected to {addr} on port")
    # 소켓 연결 종료
    sock.close()

class BoardThread(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.port = port
        self.board = None
        self.success = False

    def run(self):
        try:
            self.board = OpenBCICyton(port=self.port, daisy=False)
            self.success = True
        except Exception as e:
            print(f"Failed to initialize board on port {self.port}. Error: {e}")

#%%
import numpy as np
import matplotlib.pyplot as plt
# from pyOpenBCI import OpenBCICyton
import pickle
from datetime import datetime
# import sys
import multiprocessing
# import time
import queue
import os
# from pynput import keyboard, mouse
# import logging
# import time
# from datetime import datetime

current_path = os.getcwd()

plt.ion()

class MultiChannelEEGPlot:
    def __init__(self, queue, channels=[0, 1, 2], num_samples=2500, update_interval=25):
        self.queue = queue
        self.channels = channels
        self.num_samples = num_samples
        self.update_interval = update_interval
        self.data = {channel: np.zeros(self.num_samples) for channel in channels}
        self.fig, self.axs = plt.subplots(len(channels), 1, figsize=(10, 7))
        self.lines = {channel: ax.plot([], [], 'r-')[0] for channel, ax in zip(channels, self.axs)}
        self.start_time = time.time()  # 그래프 업데이트 시작 시간 기록
        self.queue = queue
        self.is_running = True  # 여기에서 is_running 속성을 정의
        self.update_counter = 0  # 이 카운터로 업데이트 간격을 조절
        self.downsample_factor = 2  # 다운샘플링을 위한 인자

        for ax in self.axs:
            ax.set_xlim(0, self.num_samples // self.downsample_factor)
            ax.set_ylim(-8000, 8000)

    def update_plot(self):
        sample_rate = 250  # 샘플레이트, 예를 들어 250Hz
        window_size = 5 * sample_rate  # 5초간의 데이터 수, 예: 5 * 250 = 1250
        update_y_axis_interval = 5  # y축 업데이트 간격, 초 단위
        # smoothing_window = 50  # 스무딩을 위한 데이터 윈도우 크기
        last_update_time = time.time()

        while self.is_running:
            current_time = time.time()
            try:
                data = self.queue.get_nowait()  # 큐에서 데이터 가져오기
                # 데이터를 내부 버퍼에 추가하는 코드
                for channel in self.channels:
                    self.data[channel] = np.roll(self.data[channel], -1)
                    self.data[channel][-1] = data[channel]

                self.update_counter += 1
                
                if self.update_counter >= self.update_interval:
                    for channel in self.channels:
                        downsampled_data = self.data[channel][::self.downsample_factor]
                        self.lines[channel].set_data(np.arange(len(downsampled_data)), downsampled_data)
                    
                    # 매 5초마다 y축 업데이트
                    if current_time - last_update_time >= update_y_axis_interval:
                        for channel in self.channels:
                            # 최근 5초간의 데이터 선택
                            recent_data = self.data[channel][-window_size:]
                            mean = np.mean(recent_data)
                            std = np.std(recent_data)
                            lower_bound = mean - 3*std
                            upper_bound = mean + 3*std
                            self.axs[channel].set_ylim(lower_bound, upper_bound)
                        
                        last_update_time = current_time

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    self.update_counter = 0  # 카운터 리셋

            except queue.Empty:
                time.sleep(0.01)  # 큐가 비어 있으면 잠시 대기
                continue
            
    def stop(self):
        self.is_running = False

def data_collection(queue):
    callback_count = 0  # callback 함수 호출 횟수를 저장할 변수
    full_data = []  # 모든 데이터를 누적할 리스트
    start_time = None  # 첫 callback 호출 시간
    save_interval = 5  # 데이터 저장 간격 (초)
    last_save_time = time.time()  # 마지막 저장 시간

    def callback(sample):
        nonlocal callback_count, start_time, last_save_time
        if start_time is None:
            start_time = time.time()  # 첫 callback 시간 기록

        callback_count += 1

        # FPS 계산 및 출력
        current_time = time.time()
        if current_time - start_time >= 1.0:
            print(f"FPS: {callback_count}")
            callback_count = 0
            start_time = current_time

        # 샘플 데이터를 리스트로 변환하여 누적
        current_time = time.time()
        data = [sample.channels_data[channel] for channel in range(8)] + [current_time] 
        full_data.append(data)  # 누적 데이터에 추가
        queue.put(data)  # GUI 업데이트를 위해 큐에 데이터 추가
        
        if current_time - last_save_time >= save_interval:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.pkl")
            filename = os.path.join(current_path, 'data', filename)
            with open(filename, 'wb') as file:
                pickle.dump(full_data, file)
                print(f"Data saved to {filename}")
                full_data.clear()  # 저장 후 데이터 클리어
            last_save_time = current_time

    board = OpenBCICyton(port='COM4', daisy=False)
    board.start_stream(callback)

    # 데이터 수집은 사용자가 종료할 때까지 계속 실행
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        board.stop_stream()  # 사용자 인터럽트에 의해 스트림 중지
        print("Data collection finished. Exiting program.")
        

# if __name__ == "__main__":
#     data_queue = multiprocessing.Queue()
#     data_process = multiprocessing.Process(target=data_collection, args=(data_queue,))

#     data_process.start()

#     plot = MultiChannelEEGPlot(data_queue)
#     try:
#         plot.update_plot()
#     finally:
#         plot.stop()
#         data_process.join()

#%%

def main():
    target_name = "BB24_200"  # 목표 블루투스 기기 이름
    target_address = find_bluetooth_device(target_name)
    pair_bluetooth_device(target_address)
        
    if target_address:
        print(f"Target Bluetooth device '{target_name}' found with address {target_address}")
        # 사용 가능한 모든 직렬 포트 나열
        ports = list_serial_ports()
        print("Available serial ports:", ports)
        
        for port in ports:
            print(f"Trying port {port}...")
            
            data = []
            def save_data(sample):
                data.append(sample.channels_data)  # 채널 데이터를 저장합니다.
        
            board_thread = BoardThread(port=port)
            board_thread.start()
            board_thread.join(timeout=5)  # 5초 내에 초기화되지 않으면 타임아웃
        
            if not board_thread.success:
                print(f"Port {port} failed to initialize. Moving to next port.")
                continue  # 다음 포트를 시도합니다.

            board = board_thread.board
              
            try:
                board.stop_stream()
                time.sleep(1)  # 스트리밍이 완전히 중단될 시간을 줍니다.
                print("Disconnecting the board...")
                board.disconnect()  # 보드와의 연결을 완전히 해제합니다.
                print("Board disconnected.")
                
               
                #%%
                print("Starting the data stream...")
                data_queue = multiprocessing.Queue()
                data_process = multiprocessing.Process(target=data_collection, args=(data_queue,))

                data_process.start()

                plot = MultiChannelEEGPlot(data_queue)
                try:
                    plot.update_plot()
                finally:
                    plot.stop()
                    data_process.join()
                    
                #%%

            except KeyboardInterrupt:
                print("Data collection stopped by user.")
            except Exception as e:
                print(f"Error during data collection on port {port}. Error: {e}")
            finally:
                try:
                    board.stop_stream()
                except Exception as e:
                    print(f"Failed to stop stream on port {port}. Error: {e}")

           
                #%%
if __name__ == "__main__":
    main()




























