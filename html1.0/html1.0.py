import os
import threading
import queue
from datetime import datetime, timedelta
import akshare as ak
import tushare as ts
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import volume_price_trend
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template
import re
import io
import base64
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日志队列
log_queue = queue.Queue()

# 设置Tushare Token
ts.set_token('edeb85b1c56d1e80fa3ffc550386b4547db577107e7a559ad7a89e31')
pro = ts.pro_api()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    logs = []
    image_data = None
    error = None
    # 计算默认日期
    default_date = datetime.now().strftime('%Y%m%d')

    if request.method == 'POST':
        date = request.form.get('date')
        if not re.match(r'\d{8}', date):
            error = "请输入有效日期 (YYYYMMDD)！"
            return render_template('index.html', logs=logs, image_data=image_data, error=error, default_date=default_date)

        # 清空日志队列
        while not log_queue.empty():
            log_queue.get()

        # 运行分析
        try:
            threading.Thread(
                target=run_process,
                args=(date, logs),
                daemon=True
            ).start()

            # 等待日志和图片生成
            start_time = time.time()
            while time.time() - start_time < 60:  # 最多等待60秒
                if not log_queue.empty():
                    logs.append(log_queue.get())
                if any("排序后的股票代码图片" in log for log in logs):
                    break
                time.sleep(0.1)

            # 获取图片
            output_dir = 'static'
            output_file = os.path.join(output_dir, f"stocks_{date}_sorted.png")
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

        except Exception as e:
            error = f"发生错误：{str(e)}"

    return render_template('index.html', logs=logs, image_data=image_data, error=error, default_date=default_date)

def run_process(date, logs):
    """执行分析和排序"""
    try:
        # 步骤1：获取涨停板数据
        zt_df = stock_zt_pool(date)
        if zt_df is None or zt_df.empty:
            log_queue.put("未获取到有效涨停板数据，终止分析！\n")
            return

        # 步骤2：分析股票
        analysis_data = stock_analysis(zt_df, date)

        # 步骤3：排序并生成图片
        sort_file(analysis_data, date)
        log_queue.put("分析与排序完成！\n")
    except Exception as e:
        log_queue.put(f"发生错误：{str(e)}\n")

def stock_zt_pool(date):
    """获取涨停板数据"""
    log_queue.put(f"正在获取{date}涨停板数据...\n")
    try:
        stock_zt_pool_em_df = ak.stock_zt_pool_em(date=date)
    except Exception as e:
        log_queue.put(f"获取数据失败: {str(e)}\n")
        return None

    sh_mask = stock_zt_pool_em_df['代码'].str.startswith("6") & ~stock_zt_pool_em_df['代码'].str.startswith("688")
    sz_mask = stock_zt_pool_em_df['代码'].str.startswith("0")
    mask = sh_mask | sz_mask
    filtered_df = stock_zt_pool_em_df[mask].reset_index(drop=True)

    exclude_strings = ["ST", "退", "PT", "N", "C"]
    pattern = '|'.join(exclude_strings)
    mask = ~filtered_df['名称'].str.contains(pattern, na=False)
    filtered_df = filtered_df[mask].reset_index(drop=True)

    filtered_df['主力净流入-净占比'] = 0.00
    log_queue.put(f"找到{len(filtered_df)}只有效涨停股票\n")

    for index, row in filtered_df.iterrows():
        code = row['代码']
        name = row['名称']
        log_queue.put(f"处理 [{index + 1}/{len(filtered_df)}] {code}-{name}\n")
        try:
            market = "sh" if code.startswith("6") else "sz"
            fund_flow = ak.stock_individual_fund_flow(stock=code, market=market)
            filtered_df.at[index, '主力净流入-净占比'] = fund_flow['主力净流入-净占比'].iloc[-1]
        except Exception as e:
            log_queue.put(f"  获取{code}资金流失败：{str(e)}\n")
            continue

    filtered_df['封单占成交'] = round(filtered_df['封板资金'] / filtered_df['成交额'] * 100, 2)
    filtered_df.insert(15, '换手率', filtered_df.pop('换手率'))
    sorted_df = filtered_df.sort_values(by='主力净流入-净占比', ascending=False)

    return sorted_df

def analyze_single_stock(ts_code, stock_name, start_date, end_date):
    """分析单只股票"""
    log_queue.put(f"分析股票: {ts_code}\n")
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            log_queue.put(f"{ts_code} 无数据，跳过\n")
            return None

        market_price = df['close'].sort_index(ascending=False)
        market_return = market_price.pct_change().fillna(0)
        volume = df['vol'].sort_index(ascending=False)

        rsi = RSIIndicator(close=market_price, window=14).rsi()
        macd = MACD(close=market_price).macd()
        sma50 = SMAIndicator(close=market_price, window=50).sma_indicator()
        sma200 = SMAIndicator(close=market_price, window=200).sma_indicator()
        bb = BollingerBands(close=market_price, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        prediction_strength = (rsi - 50) / 50
        prediction_strength = prediction_strength.fillna(0)
        macd_strength = (macd - macd.shift(1)).fillna(0) / 10
        trend_strength = np.where(market_price > sma50, 0.5, -0.5)
        vol_strength = np.where(volume > volume.shift(1), 0.3, -0.3)
        bb_strength = np.where(market_price > bb_upper, 0.4, np.where(market_price < bb_lower, -0.4, 0))
        vpt = volume_price_trend(close=market_price, volume=volume)
        vpt_strength = (vpt - vpt.shift(1)).fillna(0) / vpt.std() * 0.2
        prediction_strength = (
            prediction_strength * 0.4 + macd_strength * 0.2 + trend_strength * 0.2 +
            vol_strength * 0.1 + bb_strength * 0.1 + vpt_strength
        ).clip(-1, 1)

        data = pd.DataFrame({
            'date': df['trade_date'].sort_index(ascending=False),
            'prediction_strength': prediction_strength,
            'market_price': market_price,
            'market_return': market_return,
            'volume': volume,
            'sma50': sma50,
            'sma200': sma200
        })

        recent_data = data.tail(10)
        latest_pred_strength = recent_data['prediction_strength'].iloc[-1]
        if latest_pred_strength <= 0.5:
            log_queue.put(f"{ts_code} 最新预测强度 ({latest_pred_strength:.3f}) <= 0.5，跳过\n")
            return None
        if not (recent_data['prediction_strength'] > 0.5).any():
            log_queue.put(f"{ts_code} 最近10天预测强度未超过0.5，跳过\n")
            return None

        return {
            '股票代码': ts_code,
            '股票名称': stock_name,
            'prediction_strength': latest_pred_strength
        }
    except Exception as e:
        log_queue.put(f"分析 {ts_code} 失败: {str(e)}\n")
        return None

def stock_analysis(zt_df, end_date):
    """基于涨停板数据分析股票"""
    stock_codes = []
    for code in zt_df['代码'].astype(str).str.zfill(6):
        if code.startswith('60'):
            stock_codes.append(code + '.SH')
        elif code.startswith('00'):
            stock_codes.append(code + '.SZ')

    stock_name_dict = dict(zip(stock_codes, zt_df['名称']))
    all_results = []
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')

    max_workers = min(2, len(stock_codes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(
                analyze_single_stock,
                ts_code,
                stock_name_dict.get(ts_code, "未知"),
                start_date,
                end_date
            ): ts_code for ts_code in stock_codes
        }
        for future in as_completed(future_to_stock):
            result = future.result()
            if result:
                all_results.append(result)

    if not all_results:
        log_queue.put("无有效分析结果\n")
        return None

    return all_results

def sort_file(analysis_data, end_date):
    """按预测强度排序并生成图片"""
    if not analysis_data:
        log_queue.put("无分析数据可排序\n")
        return

    stock_data = []
    for item in analysis_data:
        stock_code = item['股票代码']
        if stock_code.endswith('.SH') or stock_code.endswith('.SZ'):
            stock_code = stock_code[:6]
        stock_data.append({
            'stock_code': stock_code,
            'prediction_strength': item['prediction_strength']
        })

    if not stock_data:
        log_queue.put("无有效股票代码数据\n")
        return

    stock_data_sorted = sorted(stock_data, key=lambda x: x['prediction_strength'], reverse=True)

    log_queue.put(f"按{end_date}的预测强度排序后的列表：\n")
    for item in stock_data_sorted:
        log_queue.put(f"{item['stock_code']}: {item['prediction_strength']:.3f}\n")

    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"stocks_{end_date}_sorted.png")
    create_image(stock_data_sorted, output_file, end_date)
    log_queue.put(f"排序后的股票代码图片已保存至：{output_file}\n")

def create_image(stock_data, output_path, date):
    """将股票代码列表转为图片"""
    try:
        width = 600
        line_height = 30
        padding = 15
        header_height = 50
        num_stocks = len(stock_data)
        height = header_height + num_stocks * line_height + 2 * padding

        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        title = f"排序后的股票 ({date})"
        draw.text((padding, padding), title, fill='black', font=font)

        for i, item in enumerate(stock_data):
            text = f"{i + 1}. {item['stock_code']} (强度: {item['prediction_strength']:.3f})"
            draw.text((padding, header_height + i * line_height + padding), text, fill='black', font=font)

        image.save(output_path, 'PNG')
    except Exception as e:
        log_queue.put(f"生成图片失败：{str(e)}\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)