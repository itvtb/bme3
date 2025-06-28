import serial
import matplotlib.pyplot as plt
import numpy as np

ser = serial.Serial('COM3', 115200)
plt.ion()
fig, ax = plt.subplots()

timestamps = []
values = []

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            if ',' in line:
                t, val = line.split(',')
                timestamps.append(int(t) / 1e6)  # Chuyển microsecond → giây
                values.append(int(val))

                # Giới hạn số điểm hiển thị
                if len(values) > 500:
                    timestamps.pop(0)
                    values.pop(0)

                # Vẽ đồ thị
                ax.clear()
                ax.plot(timestamps, values)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ADC Value")
                plt.pause(0.001)

except KeyboardInterrupt:
    ser.close()