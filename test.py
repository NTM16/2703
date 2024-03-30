import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

import requests
from bs4 import BeautifulSoup
import numpy as np


#Lấy danh sách mã Việt Nam
def get_stock(url):

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='wikitable collapsible autocollapse sortable')
        if table:
            stock_data = []
            # Duyệt qua mỗi hàng trong bảng
            for row in table.find_all('tr')[2:]:
                cols = row.find_all(['td', 'th'])
                if len(cols) > 0:  # Kiểm tra xem dòng có dữ liệu không
                    # Lấy dữ liệu từ mỗi ô trong hàng
                    row_data = [col.text.strip() for col in cols]
                    if row_data[2] == 'HSX':
                        stock_data.append(row_data)
                    else:
                        continue
            return stock_data
        else:
            return None
    else:
        return None

# Thay thế 'url' bằng URL thực sự mà bạn muốn lấy danh sách từ
url = 'https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_c%C3%B4ng_ty_tr%C3%AAn_s%C3%A0n_giao_d%E1%BB%8Bch_ch%E1%BB%A9ng_kho%C3%A1n_Vi%E1%BB%87t_Nam'
stock_symbols = get_stock(url)

# Lấy danh sách các mã cổ phiếu từ cột đầu tiên của bảng dữ liệu
VN_stock_list = [row[0] for row in stock_symbols]


# Tạo hộp chọn trong sidebar để chọn mã cổ phiếu
ticker = None
tick = st.sidebar.text_input('Ticker', value=None)
option_tick = st.sidebar.selectbox("Mã Việt Nam", VN_stock_list, index=None, placeholder="Chọn mã")

if not tick and not option_tick:
    ticker = None
    st.title('WELCOME TO OUR WEBSITE')
elif not tick and not option_tick:
    ticker = None
    st.title('WELCOME TO OUR WEBSITE')
elif tick and not option_tick:
    ticker = tick
elif not tick and option_tick:
    ticker = option_tick + '.VN'
elif tick and option_tick:
    ticker = None  # Reset ticker to None if both tick and option_tick have values
    st.subheader('Lưu ý chỉ chọn một mã chứng khoán')

if ticker is not None:
    caplock_ticker = ticker.title().upper()
    st.title(caplock_ticker)

    mck = yf.Ticker(ticker)

    st.subheader(mck.info['longName'])

    bsheet = mck.balance_sheet
    income = mck.income_stmt
    cfs = mck.cashflow
    statistic = mck.info
    years = bsheet.columns[-5:]  # 4 cột cho 4 năm và 1 cột cho TTM

    if bsheet.empty:
        st.caption('Không tìm thấy thông tin')

    elif income.empty:
        st.caption('Không tìm thấy thông tin')

    elif cfs.empty:
        st.caption('Không tìm thấy thông tin')

    else:
        quarter_bsheet = mck.quarterly_balance_sheet
        first_column_index = quarter_bsheet.columns[0]
        TTM_bsheet = quarter_bsheet[first_column_index]
        five_column_index = quarter_bsheet.columns[4]
        TTM_bsheet4 = quarter_bsheet[five_column_index]

        quarter_income = mck.quarterly_income_stmt
        TTM = quarter_income.iloc[:, :4].sum(axis=1)

        quarter_cfs = mck.quarterly_cashflow
        TTM_cfs = quarter_cfs.iloc[:, :4].sum(axis=1)

        summary, f_score, valuation, guru = st.tabs(
            ["Summary", "F-Score", "Valuation", "Guru"])

with guru:

    st.subheader('Liquidity Ratio')
    cr_ratio = round((TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities']), 2)

    # Lấy dữ liệu từ năm trước đến năm hiện tại
    cr_ratio_history = [bsheet.loc['Current Assets', year] / (bsheet.loc['Current Liabilities', year] or 1) for year in years[::-1]]


    # Tính toán giá trị min-max
    min_cr_ratio = round(min(cr_ratio_history), 2)
    max_cr_ratio = round(max(cr_ratio_history), 2)


    qr_ratio = round(((TTM_bsheet['Current Assets'] - TTM_bsheet['Inventory'])/TTM_bsheet['Current Liabilities']), 2) if 'Inventory' in TTM_bsheet else TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities']
    qr_ratio_history = [(bsheet.loc['Current Assets', year] - bsheet.loc['Inventory', year]) / (bsheet.loc['Current Liabilities', year] if 'Inventory' in TTM_bsheet else bsheet['Current Assets', year] / bsheet['Current Liabilities', year]) for year in years[::-1]]
    # Tính toán giá trị min-max
    min_qr_ratio = round(min(qr_ratio_history), 2)
    max_qr_ratio = round(max(qr_ratio_history), 2)

    car_ratio = round((TTM_bsheet['Cash And Cash Equivalents'] / TTM_bsheet['Current Liabilities']), 2)
    car_ratio_history = [bsheet.loc['Cash And Cash Equivalents', year] / (bsheet.loc['Current Liabilities', year] or 1) for year in years[::-1]]
    # Tính toán giá trị min-max
    min_car_ratio = round(min(car_ratio_history), 2)
    max_car_ratio = round(max(car_ratio_history), 2)

    dso_ratio = round((TTM_bsheet['Accounts Receivable'] / TTM['Total Revenue']) * 365, 2)
    dso_ratio_history = [bsheet.loc['Accounts Receivable', year] * 365 / (income.loc['Total Revenue', year] or 1) for year in years[::-1]]
    # Tính toán giá trị min-max
    min_dso_ratio = round(min(dso_ratio_history), 2)
    max_dso_ratio = round(max(dso_ratio_history), 2)

    ap_average_values = (TTM_bsheet4['Accounts Payable'] + TTM_bsheet['Accounts Payable'])/2
    dp_ratio = round((ap_average_values / TTM['Cost Of Revenue']) * 365, 2)
    dp_ratio_history = [bsheet.loc['Accounts Payable', year] * 365 / (income.loc['Cost Of Revenue', year] or 1) for year in years[::-1]]
    # Tính toán giá trị min-max
    min_dp_ratio = round(min(dp_ratio_history), 2)
    max_dp_ratio = round(max(dp_ratio_history), 2)

    inv_average = (TTM_bsheet4['Inventory'] + TTM_bsheet['Inventory'])/2 if 'Inventory' in TTM_bsheet else 0
    dio_ratio = round((inv_average / TTM['Cost Of Revenue']) * 365, 2)
    dio_ratio_history = [bsheet.loc['Inventory', year] * 365 / (income.loc['Cost Of Revenue', year] or 1) for year in years[::-1]]
    # Tính toán giá trị min-max
    min_dio_ratio = round(min(dio_ratio_history), 2)
    max_dio_ratio = round(max(dio_ratio_history), 2)

    div_ratio = round(mck.info['trailingAnnualDividendYield'] * 100, 2)
    pr_ratio = round(mck.info['payoutRatio'], 2)
    five_years_ratio = round(mck.info['fiveYearAvgDividendYield'], 2)
    forward_ratio = round(mck.info['dividendYield'] * 100, 2)
    cr_values = (cr_ratio - min_cr_ratio) / (max_cr_ratio - min_cr_ratio)
    qr_values = (qr_ratio - min_qr_ratio) / (max_qr_ratio - min_qr_ratio)
    car_values = (car_ratio - min_car_ratio) / (max_car_ratio - min_car_ratio)
    dso_values = (dso_ratio - min_dso_ratio) / (max_dso_ratio - min_dso_ratio)
    dp_values = (dp_ratio - min_dp_ratio) / (max_dp_ratio - min_dp_ratio)
    dio_values = (dio_ratio - min_dio_ratio) / (max_dio_ratio - min_dio_ratio)
    div_values = 0
    pr_values = 0
    five_years_values = 0
    forward_values = 0

    data_liquidity = pd.DataFrame(
        {
            "STT": [1, 2, 3, 4, 5, 6],
            "Index": ['Current Ratio', 'Quick Ratio', 'Cash Ratio', 'Days Inventory', 'Days Sales Outstanding',
                      'Days Payable'],
            "Current": [cr_ratio, qr_ratio, car_ratio, dio_ratio, dso_ratio, dp_ratio],
            "Vs History":[cr_values, qr_values, car_values, dio_values, dso_values, dp_values],
        }
    )
    st.data_editor(
        data_liquidity,
        column_config={
            "Vs History":st.column_config.ProgressColumn(
                "Vs History",
            ),
        },
        hide_index=True,
    )

    st.subheader('Dividend & Buy Back')
    data_dividend = pd.DataFrame(
        {
            "STT": [1, 2, 3, 4],
            "Index": ['Dividend Yield', 'Dividend Payout Ratio', '5-Year Yield-on-Cost', 'Forward Dividend Yield'],
            "Current": [div_ratio, pr_ratio, five_years_ratio, forward_ratio],
            "Vs History":[div_values, pr_values, five_years_values, forward_values],
        }
    )
    st.data_editor(
        data_dividend,
        column_config={
            "Vs History":st.column_config.ProgressColumn(
                "Vs History",
            ),
        },
        hide_index=True,
    )