import pandas as pd
import numpy as np
from scipy.stats import bernoulli

# Đọc dữ liệu từ file csv
df = pd.read_csv('/content/arima2.csv', parse_dates=['Date'], index_col='Date')

# Lấy dữ liệu của cột Value
data = df['Value'].values

# Tính xác suất p từ dữ liệu lịch sử
p = np.mean(data)

# Tạo biến ngẫu nhiên Bernoulli với xác suất p
rv = bernoulli(p)

# Dự đoán giá trị của biến Bernoulli cho một thời điểm tiếp theo
next_date = df.index[-1] + pd.DateOffset(days=1)
next_value = rv.rvs(random_state=np.random.default_rng())
# với 1 là lẻ, 0 là chẵn
# Xuất kết quả dự đoán
print(f"Giá trị của biến Bernoulli cho ngày {next_date}: {next_value}")
