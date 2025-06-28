import numpy as np

# Ví dụ kết quả vector fitting bạn cung cấp:
residues = np.array([
    -7.89384582e-09 + 1.53148096e-09j,
    1.97440741e-05 + 1.61107273e-06j,
    -3.83406454e-06 - 5.94924101e-07j,
    4.90884511e-07 + 3.60152536e-07j,
    -2.97115156e-08 - 4.88088591e-08j,
    -1.18876565e-09 - 3.46373374e-09j
])

poles = np.array([
    -6.28318531e+02 - 6.28318531e+02j,
    -6.28318531e+06 - 6.28318531e+06j,
    -9.95817762e+05 - 9.95817762e+05j,
    -1.57826479e+05 - 1.57826479e+05j,
    -2.50138112e+04 - 2.50138112e+04j,
    -3.96442192e+03 - 3.96442192e+03j
])

d = -2.851509040826747e-13 + 2.643341275032629e-13j  # điện trở thuần (gần 0)
h = -1.0000000000000007 + 2.7755575615628914e-16j  # cuộn cảm thuần (xấp xỉ -1)


# Hàm tính giá trị R, L, C từ poles và residues
def calculate_RLC(pole, residue):
    alpha = -pole.real
    omega0 = abs(pole.imag)

    # Giả sử residue thực dùng cho R, L, C (lấy phần thực để đơn giản)
    r_real = residue.real

    # Tính R, L, C gần đúng:
    R = 2 * alpha * r_real if alpha > 0 else np.nan
    L = r_real / omega0 if omega0 != 0 else np.nan
    C = 1 / (L * omega0 ** 2) if (L != 0 and omega0 != 0) else np.nan

    return R, L, C


print("Kết quả Vector Fitting phân tích thành phần mạch RLC:")

for i, (p, r) in enumerate(zip(poles, residues), 1):
    R, L, C = calculate_RLC(p, r)
    print(f"Phần tử {i}:")
    print(f"  Pole: {p}")
    print(f"  Residue: {r}")
    print(f"  => Ước lượng:")
    print(f"     R ≈ {R:.4e} Ω")
    print(f"     L ≈ {L:.4e} H")
    print(f"     C ≈ {C:.4e} F")
    print("")

print(f"Điện trở thuần (d): {d.real:.4e} Ω")
print(f"Cuộn cảm thuần (h): {h.real:.4e} H (âm tính có thể cần xét lại dấu hoặc mô hình)")

