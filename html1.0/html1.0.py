import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import volume_price_trend
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template
import re
import io
import base64
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    logs = []
    image_data = None
    error = None
    default_date = datetime.now().strftime('%Y%m%d')

    if request.method == 'POST':
        date = request.form.get('date')
        if not re.match(r'\d{8}', date):
            error = "请输入有效日期 (YYYYMMDD)！"
            return render_template('index.html', logs=logs, image_data=image_data, error=error, default_date=date)

        try:
            zt_df = stock_zt_pool(date)
            if zt_df is None or zt_df.empty:
                logs.append("未获取到有效涨停板数据，终止分析！")
                return render_template('index.html', logs=logs, image_data=image_data, error=error, default_date=date)

            analysis_data = stock_analysis(zt_df, date)
            image_data = sort_file(analysis_data, date)
            logs.append("分析与排序完成！")

        except Exception as e:
            error = f"发生错误：{str(e)}"

    return render_template('index.html', logs=logs, image_data=image_data, error=error, default_date=default_date)

def stock_zt_pool(date):
    """获取涨停板数据（使用 akshare）"""
    logger.info(f"获取{date}涨停板数据...")
    try:
        df = ak.stock_zt_pool(date=date)
        if df is None or df.empty:
            logger.warning(f"{date} 无涨停板数据")
            return pd.DataFrame()
        # 重命名列以匹配分析逻辑
        df = df.rename(columns={'股票代码': '代码', '股票名称': '名称'})
        return df[['代码', '名称', '封板资金', '成交额', '换手率']].dropna()
    except Exception as e:
        logger.error(f"获取涨停板数据失败：{str(e)}")
        return pd.DataFrame()

def analyze_single_stock(ts_code, stock_name, start_date, end_date):
    """分析单只股票（使用 akshare 获取历史数据）"""
    logger.info(f"分析股票: {ts_code}")
    try:
        # 获取历史数据
        df = ak.stock_zh_a_hist(symbol=ts_code[:6], start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            logger.warning(f"{ts_code} 无历史数据")
            return None

        market_price = df['收盘'].iloc[::-1]
        volume = df['成交量'].iloc[::-1]

        rsi = RSIIndicator(close=market_price, window=14).rsi()
        macd = MACD(close=market_price).macd()
        sma50 = SMAIndicator(close=market_price, window=50).sma_indicator()
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

        latest_pred_strength = prediction_strength.iloc[-1]
        if latest_pred_strength <= 0.5:
            logger.info(f"{ts_code} 最新预测强度 ({latest_pred_strength:.3f}) <= 0.5，跳过")
            return None

        return {
            '股票代码': ts_code,
            '股票名称': stock_name,
            'prediction_strength': latest_pred_strength
        }
    except Exception as e:
        logger.error(f"分析{ts_code}失败：{str(e)}")
        return None

def stock_analysis(zt_df, end_date):
    """分析股票"""
    stock_codes = [code + '.SH' if code.startswith('6') else code + '.SZ' for code in zt_df['代码']]
    stock_name_dict = dict(zip(stock_codes, zt_df['名称']))
    all_results = []
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')

    for ts_code in stock_codes:
        result = analyze_single_stock(ts_code, stock_name_dict.get(ts_code, '未知'), start_date, end_date)
        if result:
            all_results.append(result)

    if not all_results:
        logger.info("无有效分析结果")
        return None
    return all_results

def sort_file(analysis_data, end_date):
    """排序并生成图片"""
    if not analysis_data:
        logger.info("无分析数据可排序")
        return None

    stock_data = [{'stock_code': item['股票代码'][:6], 'prediction_strength': item['prediction_strength']} for item in analysis_data]
    stock_data_sorted = sorted(stock_data, key=lambda x: x['prediction_strength'], reverse=True)

    logger.info(f"按{end_date}的预测强度排序后的列表：")
    for item in stock_data_sorted:
        logger.info(f"{item['stock_code']}: {item['prediction_strength']:.3f}")

    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'stocks_{end_date}_sorted.png')
    create_image(stock_data_sorted, output_file, end_date)
    with open(output_file, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_image(stock_data, output_path, date):
    """生成图片"""
    try:
        width = 600
        line_height = 30
        padding = 15
        header_height = 50
        num_stocks = len(stock_data)
        height = header_height + num_stocks * line_height + 2 * padding

        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        title = f"排序后的股票 ({date})"
        draw.text((padding, padding), title, fill='black', font=font)

        for i, item in enumerate(stock_data):
            text = f"{i + 1}. {item['stock_code']} (强度: {item['prediction_strength']:.3f})"
            draw.text((padding, header_height + i * line_height + padding), text, fill='black', font=font)

        image.save(output_file, 'PNG')
    except Exception as e:
        logger.error(f"生成图片失败：{str(e)}")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))