import os
import time
from datetime import datetime, timedelta
import akshare as ak
import tushare as ts
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import volume_price_trend
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template, send_file, jsonify
from kivy.logger import Logger
import queue

# 日志队列
log_queue = queue.Queue()

# 设置 Tushare Token
ts.set_token('edeb85b1c56d1e80fa3ffc550386b4547db577107e7a559ad7a89e31')
pro = ts.pro_api()

app = Flask(__name__)

# 复制 main.py 中的函数
def stock_zt_pool(save_directory, date):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        log_queue.put(f"已创建目录：{save_directory}\n")

    log_queue.put(f"\n正在获取{date}涨停板数据...\n")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock_zt_pool_em_df = ak.stock_zt_pool_em(date=date)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                log_queue.put(f"获取数据失败 (尝试 {attempt + 1}/{max_retries}): {e}，重试中...\n")
                time.sleep(2)
            else:
                log_queue.put(f"获取数据失败: {e}\n")
                return None

    sh_mask = stock_zt_pool_em_df['代码'].str.startswith("6") & ~stock_zt_pool_em_df['代码'].str.startswith("688")
    sz_mask = stock_zt_pool_em_df['代码'].str.startswith("0")
    mask = sh_mask | sz_mask
    filtered_df = stock_zt_pool_em_df[mask].reset_index(drop=True)

    exclude_strings = ["ST", "退", "PT", "N", "C"]
    pattern = '|'.join(exclude_strings)
    mask = ~filtered_df['名称'].str.contains(pattern)
    filtered_df = filtered_df[mask].reset_index(drop=True)

    filtered_df['主力净流入-净占比'] = 0.00
    log_queue.put(f"找到{len(filtered_df)}只有效涨停股票\n")

    for index, row in filtered_df.iterrows():
        code = row['代码']
        name = row['名称']
        log_queue.put(f"正在处理 [{index + 1}/{len(filtered_df)}] {code}-{name}\n")
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
    log_queue.put(f"正在分析股票: {ts_code}\n")
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
            log_queue.put(f"{ts_code} 最新一天 prediction_strength ({latest_pred_strength:.3f}) <= 0.5，跳过\n")
            return None
        if not (recent_data['prediction_strength'] > 0.5).any():
            log_queue.put(f"{ts_code} 最近10天 prediction_strength 未超过0.5，跳过\n")
            return None

        latest_date = data['date'].iloc[0]
        latest_data = data.iloc[0]
        prev_data = data.iloc[1] if len(data) > 1 else None

        sma5 = SMAIndicator(close=market_price, window=5).sma_indicator()
        trend_reversal = (sma5.shift(1) > sma50.shift(1)) & (sma5 < sma50)

        atr = AverageTrueRange(
            high=df['high'].sort_index(ascending=False),
            low=df['low'].sort_index(ascending=False),
            close=market_price,
            window=14
        ).average_true_range()
        dynamic_threshold_low_return = -atr.iloc[0] * 0.5 / 100
        dynamic_threshold_high_return = atr.iloc[0] * 0.5 / 100

        threshold_high_pred = 0.4
        threshold_low_pred = -0.4
        trend_filter = latest_data['market_price'] > latest_data['sma50']
        recent_highs = market_price.rolling(window=5).max()
        is_near_high = latest_data['market_price'] > recent_highs.iloc[0] * 0.95

        buy_signal = (
            latest_pred_strength > threshold_high_pred and
            latest_data['market_return'] < dynamic_threshold_low_return and
            trend_filter and
            latest_data['volume'] > prev_data['volume'] * 0.9 and
            not trend_reversal.iloc[0] and
            not is_near_high
        )
        sell_signal = (
            latest_pred_strength < threshold_low_pred and
            latest_data['market_return'] > dynamic_threshold_high_return and
            not trend_filter and
            latest_data['volume'] > prev_data['volume'] * 0.9 and
            not trend_reversal.iloc[0]
        )

        log_queue.put(f"{ts_code} ({stock_name}) 最新交易日（{end_date}）分析:\n")
        log_queue.put(f"预测强度: {latest_pred_strength:.3f}\n")
        signal = "持仓: 无明确信号。"
        if buy_signal:
            signal = "买入信号: 建议下一交易日买入！"
        elif sell_signal:
            signal = "卖出信号: 建议下一交易日卖出！"
        log_queue.put(signal + "\n")

        recent_buy_signals = (
            (recent_data['prediction_strength'] > threshold_high_pred) &
            (recent_data['market_return'] < dynamic_threshold_low_return) &
            (recent_data['market_price'] > recent_data['sma50']) &
            (recent_data['volume'] > recent_data['volume'].shift(1) * 0.9) &
            (~trend_reversal.tail(10))
        )
        recent_sell_signals = (
            (recent_data['prediction_strength'] < threshold_low_pred) &
            (recent_data['market_return'] > dynamic_threshold_high_return) &
            (recent_data['market_price'] <= recent_data['sma50']) &
            (recent_data['volume'] > recent_data['volume'].shift(1) * 0.9) &
            (~trend_reversal.tail(10))
        )

        prediction_strength_avg = recent_data['prediction_strength'].mean()

        return {
            '股票代码': ts_code,
            '股票名称': stock_name,
            '最近10天数据': recent_data[['date', 'prediction_strength', 'market_price', 'market_return', 'volume']].to_string(index=False),
            'prediction_strength 平均值': prediction_strength_avg,
            '买入信号日期': recent_data['date'][recent_buy_signals].tolist(),
            '卖出信号日期': recent_data['date'][recent_sell_signals].tolist(),
            '信号': signal
        }
    except Exception as e:
        log_queue.put(f"分析 {ts_code} 失败: {str(e)}\n")
        return None

def stock_analysis(zt_df, end_date, save_directory):
    stock_codes = []
    for code in zt_df['代码'].astype(str).str.zfill(6):
        if code.startswith('60'):
            stock_codes.append(code + '.SH')
        elif code.startswith('00'):
            stock_codes.append(code + '.SZ')
        else:
            log_queue.put(f"跳过不支持的代码格式: {code}\n")

    stock_name_dict = dict(zip(stock_codes, zt_df['名称']))
    all_results = []
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')

    max_workers = min(4, len(stock_codes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(
                analyze_single_stock,
                ts_code,
                stock_name_dict.get(ts_code, "未知名称"),
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

    file_name = f"股票分析结果_{end_date}.txt"
    output_file = os.path.join(save_directory, file_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(f"股票代码: {result['股票代码']}\n")
            f.write(f"股票名称: {result['股票名称']}\n")
            f.write("最近10天数据:\n")
            f.write(f"{result['最近10天数据']}\n")
            f.write(f"prediction_strength 平均值: {result['prediction_strength 平均值']}\n")
            f.write("买入信号日期: " + str(result['买入信号日期']) + "\n")
            f.write("卖出信号日期: " + str(result['卖出信号日期']) + "\n")
            f.write(f"信号: {result['信号']}\n")
            f.write("-" * 50 + "\n\n")

    log_queue.put(f"分析完成，结果已保存至 {output_file}\n")
    return output_file

def generate_screenshot(file_path, output_image_path):
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read()
        lines = content.split('\n')
        font_size = 20
        line_height = font_size + 5
        max_width = 0
        try:
            font = ImageFont.truetype("simsun.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("DroidSansFallback.ttf", font_size)
            except:
                font = ImageFont.load_default()
                log_queue.put("警告：未找到合适的字体，使用默认字体，可能影响中文显示\n")

        draw = ImageDraw.Draw(Image.new('RGB', (1, 1), 'white'))
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            max_width = max(max_width, text_width)

        image_width = max_width + 20
        image_height = len(lines) * line_height + 20

        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)
        y = 10
        for line in lines:
            draw.text((10, y), line, font=font, fill='black')
            y += line_height

        image.save(output_image_path, 'PNG')
        log_queue.put(f"截图已保存至: {output_image_path}\n")
        return output_image_path
    except Exception as e:
        log_queue.put(f"生成截图失败: {str(e)}\n")
        return None

def sort_file(input_file, output_dir, end_date):
    log_queue.put(f"开始排序文件: {input_file}\n")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('-' * 50 + '\n\n')
    blocks = [block.strip() for block in blocks if block.strip()]
    if not blocks:
        raise ValueError("文件内容为空或格式不正确")

    stock_data = []
    for block in blocks:
        lines = block.split('\n')
        stock_code = None
        stock_name = None
        for line in lines:
            if line.startswith('股票代码'):
                stock_code = line.replace('股票代码: ', '').strip()
            if line.startswith('股票名称'):
                stock_name = line.replace('股票名称: ', '').strip()

        if not stock_code or not stock_name:
            log_queue.put(f"块解析失败，跳过：\n{block}\n")
            continue

        if stock_code.startswith(('SH', 'SZ', 'sh', 'sz')) and len(stock_code) == 9:
            stock_code = stock_code[2:]
        elif stock_code.endswith(('.SH', '.SZ', '.sh', '.sz')):
            stock_code = re.sub(r'\.(SH|SZ|sh|sz)$', '', stock_code)
        elif not stock_code.isdigit() or len(stock_code) not in (6, 9):
            log_queue.put(f"股票代码格式不正确，跳过：{stock_code}\n")
            continue

        data_section = False
        data_lines = []
        for line in lines:
            if '最近10天数据' in line:
                data_section = True
                continue
            if data_section and line.startswith('买入信号日期'):
                break
            if data_section:
                data_lines.append(line)

        dates = [line.split()[0] for line in data_lines[1:] if line.split()]
        sort_date = end_date if end_date in dates else dates[-1] if dates else None
        if sort_date is None:
            log_queue.put(f"{stock_code} 数据中没有有效的日期，跳过\n")
            continue

        if sort_date != end_date:
            log_queue.put(f"{stock_code} 未找到 {end_date}，使用最后一天 {sort_date}\n")

        prediction_strength = None
        for line in data_lines[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == sort_date:
                try:
                    prediction_strength = float(parts[1])
                    break
                except ValueError:
                    log_queue.put(f"无法解析 {stock_code} 的 prediction_strength {parts[1]}\n")
                    continue

        if prediction_strength is None:
            log_queue.put(f"{stock_code} 未找到 {sort_date} 的 prediction_strength，跳过\n")
            continue

        stock_data.append({
            'block': block,
            'prediction_strength': prediction_strength,
            'stock_code': stock_code
        })

    if not stock_data:
        raise ValueError("没有找到任何个股的 prediction_strength 数据")

    stock_data_sorted = sorted(stock_data, key=lambda x: x['prediction_strength'], reverse=True)

    log_queue.put(f"按 {sort_date} 的 prediction_strength 排序后的列表：\n")
    for item in stock_data_sorted:
        log_queue.put(f"{item['stock_code']}: {item['prediction_strength']:.3f}\n")

    output_file = os.path.join(output_dir, f"股票分析结果_{end_date}_排序后.txt")
    codes_only_file = os.path.join(output_dir, f"股票代码_{end_date}_排序后.txt")

    with open(output_file, 'w', encoding='gbk', newline='') as f:
        for i, item in enumerate(stock_data_sorted):
            f.write(item['block'])
            f.write('\n')
            if i < len(stock_data_sorted) - 1:
                f.write('-' * 50 + '\n\n')

    with open(codes_only_file, 'w', encoding='gbk', newline='') as f:
        f.write("代码\n")
        for item in stock_data_sorted:
            f.write(f"{item['stock_code']}\n")

    log_queue.put(f"排序后的详细文件已保存至：{output_file}\n")
    log_queue.put(f"排序后的股票代码文件已保存至：{codes_only_file}\n")

    screenshot_path = os.path.join(output_dir, f"股票代码_{end_date}_截图.png")
    generate_screenshot(codes_only_file, screenshot_path)
    return screenshot_path

# Web 路由
@app.route('/')
def index():
    return render_template('index.html', default_date=datetime.today().strftime('%Y-%m-%d'))

@app.route('/analyze', methods=['POST'])
def analyze():
    date_str = request.form['date']
    save_directory = os.path.join(os.getcwd(), "stock_output")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y%m%d')
        zt_df = stock_zt_pool(save_directory, date)
        if zt_df is None or zt_df.empty:
            return jsonify({'error': '未获取到有效涨停板数据，终止分析！'})

        analysis_file = stock_analysis(zt_df, date, save_directory)
        screenshot_path = sort_file(analysis_file, save_directory, date)
        if os.path.exists(screenshot_path):
            return send_file(screenshot_path, as_attachment=True, download_name=f"stock_analysis_{date}.png")
        else:
            return jsonify({'error': '生成截图失败！'})
    except Exception as e:
        log_queue.put(f"\n发生错误：{str(e)}")
        return jsonify({'error': str(e)})

@app.route('/logs')
def get_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
