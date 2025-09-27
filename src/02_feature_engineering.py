"""
Human vs AI Generated Essays 数据分析项目
02 - 特征工程

作者：bosprimigenious
生成日期：2025-9-26-15:32
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# NLP相关库
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
except ImportError:
    print("警告: NLTK 未安装，某些功能可能不可用")
    nltk = None

try:
    from textstat import textstat
except ImportError:
    print("警告: textstat 未安装，可读性指标功能不可用")
    textstat = None

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 导入图表样式配置和结果导出
from plot_style import setup_plot_style, create_beautiful_plot, save_beautiful_plot, apply_theme
from results_exporter import ResultsExporter

# 设置图表样式
setup_plot_style()
apply_theme('professional')

def download_nltk_data():
    """下载必要的NLTK数据"""
    if nltk is None:
        return
    
    try:
        print("正在下载NLTK数据...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)  # 添加这个资源
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        print("✓ NLTK 数据下载完成")
    except Exception as e:
        print(f"⚠ NLTK 数据下载失败: {e}")
        print("某些高级特征可能不可用，但程序会继续运行")

def load_processed_data():
    """加载处理后的数据"""
    data_path = os.path.join('data', 'processed_data.csv')
    
    if not os.path.exists(data_path):
        print(f"处理后的数据文件不存在: {data_path}")
        print("请先运行 01_data_exploration.py")
        return None
    
    print(f"正在加载处理后的数据: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"数据加载完成，形状: {df.shape}")
    
    return df

def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ""
    
    # 转小写
    text = str(text).lower()
    
    # 移除特殊字符和数字，保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_text_preprocessing(text):
    """高级文本预处理（需要NLTK）"""
    if nltk is None:
        return preprocess_text(text)
    
    if pd.isna(text):
        return ""
    
    # 基础预处理
    text = preprocess_text(text)
    
    try:
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"高级预处理失败，使用基础预处理: {e}")
        return preprocess_text(text)

def extract_basic_features(text):
    """提取基础文本特征"""
    if pd.isna(text):
        return {}
    
    text = str(text)
    
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        'paragraph_count': len([p for p in text.split('\n') if p.strip()]),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'avg_sentence_length': len(text.split()) / max(len([s for s in text.split('.') if s.strip()]), 1),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'comma_count': text.count(','),
        'semicolon_count': text.count(';'),
        'colon_count': text.count(':'),
        'quotation_count': text.count('"') + text.count("'"),
        'capital_letter_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        'space_ratio': sum(1 for c in text if c.isspace()) / max(len(text), 1),
    }
    
    return features

def extract_readability_features(text):
    """提取可读性特征（需要textstat）"""
    if textstat is None or pd.isna(text):
        return {}
    
    try:
        text = str(text)
        features = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        }
        return features
    except Exception as e:
        print(f"可读性特征提取失败: {e}")
        return {}

# 全局变量跟踪词性特征提取状态
_pos_feature_error_logged = False

def extract_pos_features(text):
    """提取词性标注特征（需要NLTK）"""
    global _pos_feature_error_logged
    
    if nltk is None or pd.isna(text):
        return {}
    
    try:
        text = str(text)
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # 统计各种词性
        pos_counts = Counter([tag for _, tag in pos_tags])
        
        features = {
            'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                         pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0),
            'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                         pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                         pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),
            'adjective_count': pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),
            'adverb_count': pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0),
            'pronoun_count': pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0),
            'determiner_count': pos_counts.get('DT', 0),
            'preposition_count': pos_counts.get('IN', 0),
            'conjunction_count': pos_counts.get('CC', 0),
        }
        
        # 计算比例
        total_tokens = len(tokens)
        if total_tokens > 0:
            for key in features:
                features[key + '_ratio'] = features[key] / total_tokens
        
        return features
    except Exception as e:
        # 只记录一次错误，避免重复输出
        if not _pos_feature_error_logged:
            print(f"⚠ 词性特征提取失败: {e}")
            print("   将跳过词性特征，继续其他特征提取...")
            _pos_feature_error_logged = True
        return {}

def extract_ngram_features(text, n=2):
    """提取N-gram特征"""
    if pd.isna(text):
        return {}
    
    text = str(text).lower()
    tokens = text.split()
    
    if len(tokens) < n:
        return {}
    
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_counts = Counter(ngrams)
    
    features = {
        f'{n}gram_count': len(ngrams),
        f'{n}gram_unique_count': len(ngram_counts),
        f'{n}gram_diversity': len(ngram_counts) / max(len(ngrams), 1),
    }
    
    return features

def extract_vocabulary_features(text):
    """提取词汇特征"""
    if pd.isna(text):
        return {}
    
    text = str(text).lower()
    words = text.split()
    
    if not words:
        return {}
    
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    
    # 计算词汇多样性指标
    features = {
        'vocabulary_size': unique_words,
        'type_token_ratio': unique_words / total_words,
        'hapax_legomena': sum(1 for count in word_counts.values() if count == 1),
        'hapax_ratio': sum(1 for count in word_counts.values() if count == 1) / unique_words,
        'dis_legomena': sum(1 for count in word_counts.values() if count == 2),
        'dis_ratio': sum(1 for count in word_counts.values() if count == 2) / unique_words,
    }
    
    return features

def extract_all_features(df):
    """提取所有特征"""
    print("\n" + "="*50)
    print("开始特征提取")
    print("="*50)
    
    # 下载NLTK数据
    download_nltk_data()
    
    # 文本预处理
    print("1. 文本预处理...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df['advanced_cleaned_text'] = df['text'].apply(advanced_text_preprocessing)
    print(f"   ✓ 预处理完成，处理了 {len(df)} 个文本")
    
    # 基础特征
    print("2. 提取基础特征...")
    basic_features = df['text'].apply(extract_basic_features)
    basic_df = pd.DataFrame(basic_features.tolist())
    print(f"   ✓ 基础特征提取完成，特征数: {basic_df.shape[1]}")
    
    # 可读性特征
    print("3. 提取可读性特征...")
    readability_features = df['text'].apply(extract_readability_features)
    readability_df = pd.DataFrame(readability_features.tolist())
    print(f"   ✓ 可读性特征提取完成，特征数: {readability_df.shape[1]}")
    
    # 词性特征
    print("4. 提取词性特征...")
    pos_features = df['text'].apply(extract_pos_features)
    pos_df = pd.DataFrame(pos_features.tolist())
    print(f"   ✓ 词性特征提取完成，特征数: {pos_df.shape[1]}")
    
    # N-gram特征
    print("5. 提取N-gram特征...")
    bigram_features = df['text'].apply(lambda x: extract_ngram_features(x, 2))
    bigram_df = pd.DataFrame(bigram_features.tolist())
    
    trigram_features = df['text'].apply(lambda x: extract_ngram_features(x, 3))
    trigram_df = pd.DataFrame(trigram_features.tolist())
    print(f"   ✓ N-gram特征提取完成，2-gram: {bigram_df.shape[1]}, 3-gram: {trigram_df.shape[1]}")
    
    # 词汇特征
    print("6. 提取词汇特征...")
    vocab_features = df['cleaned_text'].apply(extract_vocabulary_features)
    vocab_df = pd.DataFrame(vocab_features.tolist())
    print(f"   ✓ 词汇特征提取完成，特征数: {vocab_df.shape[1]}")
    
    # 合并所有特征
    print("7. 合并特征...")
    feature_dfs = [basic_df, readability_df, pos_df, bigram_df, trigram_df, vocab_df]
    
    # 处理缺失值
    for i, feature_df in enumerate(feature_dfs):
        feature_df.fillna(0, inplace=True)
        print(f"   ✓ 特征组 {i+1} 缺失值处理完成")
    
    # 合并特征
    all_features = pd.concat([df] + feature_dfs, axis=1)
    
    # 统计特征信息
    original_cols = len(df.columns)
    new_feature_cols = all_features.shape[1] - original_cols
    
    print(f"\n特征提取完成！")
    print(f"   - 原始列数: {original_cols}")
    print(f"   - 新增特征数: {new_feature_cols}")
    print(f"   - 总特征数: {all_features.shape[1]}")
    print(f"   - 数据形状: {all_features.shape}")
    
    return all_features

def create_tfidf_features(df, max_features=5000):
    """创建TF-IDF特征"""
    print("\n" + "="*50)
    print("创建TF-IDF特征")
    print("="*50)
    
    # 使用清理后的文本
    texts = df['advanced_cleaned_text'].fillna('')
    
    # TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    print("拟合TF-IDF向量化器...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 获取特征名称
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print(f"TF-IDF特征矩阵形状: {tfidf_matrix.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 保存向量化器
    vectorizer_path = os.path.join('data', 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF向量化器已保存到: {vectorizer_path}")
    
    return tfidf_matrix, feature_names

def analyze_feature_importance(df):
    """分析特征重要性"""
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)
    
    # 选择数值特征
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['generated', 'text_length', 'word_count']]
    
    print(f"数值特征数量: {len(numeric_features)}")
    
    # 计算特征与标签的相关性
    correlations = df[numeric_features].corrwith(df['generated']).abs().sort_values(ascending=False)
    
    print("\n与标签相关性最高的特征:")
    print(correlations.head(20))
    
    # 可视化特征相关性
    fig, ax, colors = create_beautiful_plot(figsize=(12, 8), 
                                           title='特征重要性分析 - 与标签的相关性',
                                           xlabel='与标签的绝对相关性',
                                           ylabel='特征名称')
    
    top_features = correlations.head(20)
    bars = ax.barh(range(len(top_features)), top_features.values, 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    save_beautiful_plot(fig, 'feature_importance.png')
    plt.show()
    
    return correlations

def save_features(df, tfidf_matrix, feature_names):
    """保存特征数据"""
    print("\n" + "="*50)
    print("保存特征数据")
    print("="*50)
    
    # 保存处理后的数据
    processed_path = os.path.join('data', 'features_data.csv')
    df.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"特征数据已保存到: {processed_path}")
    
    # 保存TF-IDF矩阵
    tfidf_path = os.path.join('data', 'tfidf_matrix.npz')
    np.savez_compressed(tfidf_path, matrix=tfidf_matrix.toarray())
    print(f"TF-IDF矩阵已保存到: {tfidf_path}")
    
    # 保存特征名称
    feature_names_path = os.path.join('data', 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"特征名称已保存到: {feature_names_path}")

def main():
    """主函数"""
    print("Human vs AI Generated Essays 数据分析项目")
    print("02 - 特征工程")
    print("="*50)
    
    try:
        # 加载数据
        df = load_processed_data()
        if df is None:
            return False
        
        # 提取所有特征
        df_with_features = extract_all_features(df)
        
        # 创建TF-IDF特征
        tfidf_matrix, feature_names = create_tfidf_features(df_with_features)
        
        # 分析特征重要性
        correlations = analyze_feature_importance(df_with_features)
        
        # 导出特征工程结果
        print("\n" + "="*50)
        print("导出特征工程结果")
        print("="*50)
        exporter = ResultsExporter()
        exporter.export_feature_engineering_results(df_with_features, correlations)
        
        # 保存特征数据
        save_features(df_with_features, tfidf_matrix, feature_names)
        
        print("\n" + "="*50)
        print("特征工程完成！")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
