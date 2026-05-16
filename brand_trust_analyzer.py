import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# 1. INITIALIZE VADER
analyzer = SentimentIntensityAnalyzer()

# 2. FAIL-SAFE EMOTION & TRUST LOGIC
def advanced_marketing_analysis(text):
    if not isinstance(text, str) or text.strip() == '':
        return pd.Series([0.0, "Neutral", "No Review"])
    
    text_lower = text.lower()
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    
    emotions = {
        'Anger': ['angry', 'mad', 'annoyed', 'furious', 'scam', 'hate', 'lied'],
        'Fear': ['scared', 'worried', 'threat', 'data', 'privacy', 'risk', 'stole'],
        'Sadness': ['disappointed', 'sad', 'regret', 'unfortunate', 'poor', 'lost'],
        'Trust': ['reliable', 'honest', 'good', 'excellent', 'safe', 'helpful']
    }
    
    primary_emotion = "Neutral"
    for emotion, keywords in emotions.items():
        if any(word in text_lower for word in keywords):
            primary_emotion = emotion
            break
            
    category = "Neutral/Positive"
    if compound < -0.3:
        integrity_keywords = ['lie', 'cheat', 'scam', 'fake', 'hidden', 'stole', 'fraud', 'unauthorized']
        competence_keywords = ['broke', 'slow', 'useless', 'fail', 'bad quality', 'error', 'broken', 'bug']
        
        if any(w in text_lower for w in integrity_keywords):
            category = "Integrity Violation"
        elif any(w in text_lower for w in competence_keywords):
            category = "Competence Violation"
        else:
            category = "General Dissatisfaction"
            
    return pd.Series([compound, primary_emotion, category])

# 3. CREATE RESEARCH-READY SYNTHETIC DATASET
# Based on real CFPB complaint patterns across financial products
synthetic_complaints = [
    # Integrity Violations - Hidden Fees & Unauthorized Charges
    "They charged me a hidden monthly fee without notification. When I called, they lied about the terms.",
    "Bank opened three accounts in my name without authorization. Complete fraud and identity misuse!",
    "Hidden fees appeared on my statement that were never disclosed. They stole money from my account.",
    "They promised no fees for the first year but charged me $150. I have the signed agreement as proof.",
    "Unauthorized credit card was issued under my name. Nobody at the bank will take responsibility.",
    "They lied about the interest rate and now I'm paying 29.99% instead of the promised 15%.",
    "Scam alert! They keep charging me for insurance I never agreed to purchase. Been going on for months.",
    "Account was charged twice for the same transaction. Bank refuses to reverse the duplicate charge.",
    "They added unwanted products to my loan without my consent. Pure fraud and deception.",
    
    # Competence Violations - System Failures & Poor Quality
    "Mobile app crashes every time I try to transfer money. Unusable product, been broken for weeks.",
    "Online banking portal is completely broken. Can't check my balance for the past three days.",
    "Their ATM failed during deposit and didn't credit my account. Slow response from customer service.",
    "The new update is full of bugs. Payment failed multiple times due to system error.",
    "Website has been slow and useless for weeks. Can't manage my account properly.",
    "Interest was calculated incorrectly for six months. Major accounting error on their part.",
    "Card reader is broken at multiple branches. Bad quality equipment and poor maintenance.",
    "Transaction failed but money was deducted. Their system is buggy and unreliable.",
    "App keeps logging me out randomly. Slow and broken functionality since the last update.",
    
    # Mixed - Late Responses & Poor Handling
    "Submitted dispute 45 days ago but no response. They keep saying it's under review.",
    "Customer service promised a callback within 24 hours but never called. This happened three times.",
    "Poor communication about account changes. Fees increased without proper notification.",
    "They lost my paperwork twice. Now they claim I missed the deadline for my dispute.",
    
    # General Dissatisfaction
    "Interest rates are too high compared to other banks. Very disappointed with their rates.",
    "Branch locations are inconvenient and the hours don't work for working professionals.",
    "Got a better offer from another bank. Current bank won't match despite 10 years of loyalty.",
    "Limited ATM network in my area. Have to pay fees to access my own money.",
    
    # Positive/Negative Mixed
    "The app is good and reliable, but customer service was rude when I called about a fee.",
    "Excellent security features but the interest rates are disappointing compared to competitors.",
    "Safe and reliable bank but they made an error on my mortgage calculation.",
    
    # Strong Positive (for contrast)
    "Excellent customer service! They helped me resolve a dispute quickly and professionally.",
    "This bank is honest and transparent about all fees. Highly recommended for reliability.",
    "Safe and secure banking experience. The app is reliable and easy to use.",
    "Good experience overall. They were helpful when I had questions about my account.",
]

# Create DataFrame
df = pd.DataFrame({'review': synthetic_complaints}) 
df.index += 1

# Add some realistic metadata for richer analysis
np.random.seed(42)
companies = ['JPMorgan Chase', 'Bank of America', 'Wells Fargo', 'Citibank', 
            'US Bank', 'PNC Bank', 'TD Bank', 'Capital One']
products = ['Checking account', 'Credit card', 'Mortgage', 'Auto loan', 
            'Savings account', 'Personal loan']

df['Company'] = np.random.choice(companies, len(df))
df['Product'] = np.random.choice(products, len(df))
df['Review_Length'] = df['review'].str.len()

print(f"Created synthetic dataset with {len(df)} realistic complaint narratives")
print("Complaint types covered: Integrity Violations, Competence Violations, General Dissatisfaction\n")

# 4. EXECUTE ANALYSIS
analysis_results = df['review'].apply(advanced_marketing_analysis)

# Convert DataFrame to separate columns directly
df['Sentiment'] = analysis_results[0]
df['Primary_Emotion'] = analysis_results[1]
df['Trust_Category'] = analysis_results[2]
print("Sample Analysis Results:")
print(df[['review', 'Trust_Category', 'Primary_Emotion', 'Sentiment']]) 
print("\n")

# 5. COMPREHENSIVE VISUALIZATION DASHBOARD
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 14), constrained_layout=0)

# More spacing between plots
gs = fig.add_gridspec(
    2, 3,
    hspace=0.70,
    wspace=0.60,
    top=0.85
)



# CONSISTENT COLOR SYSTEM

colors_dict = {
    'Integrity Violation': '#E63946',        # red
    'Competence Violation': '#457B9D',       # blue
    'General Dissatisfaction': '#F4A261',    # orange
    'Neutral/Positive': '#2A9D8F'            # green
}

emotion_colors = {
    'Anger': '#E63946',
    'Fear': '#6A4C93',
    'Sadness': '#457B9D',
    'Trust': '#2A9D8F',
    'Neutral': '#A8DADC'
}


# CHART 1
ax1 = fig.add_subplot(gs[0, 0])

category_order = [
    'Integrity Violation',
    'Competence Violation',
    'General Dissatisfaction',
    'Neutral/Positive'
]

category_counts = (
    df['Trust_Category']
    .value_counts()
    .reindex(category_order)
)

bar_colors = [
    colors_dict[cat]
    for cat in category_counts.index
]

ax1.bar(
    category_counts.index,
    category_counts.values,
    color=bar_colors,
    edgecolor='black'
)

ax1.set_title(
    'Brand Trust Violation Categories',
    fontweight='bold',
    fontsize=13,
    pad=15
)

ax1.set_ylabel('Number of Complaints')
ax1.tick_params(axis='x', rotation=30, pad=8)

for i, v in enumerate(category_counts.values):
    ax1.text(i, v + 0.3, str(v), ha='center', fontweight='bold')


# CHART 2
ax2 = fig.add_subplot(gs[0, 1])

emotion_cross = pd.crosstab(
    df['Trust_Category'],
    df['Primary_Emotion']
)

emotion_cross = emotion_cross.reindex(category_order)

emotion_cross.plot(
    kind='bar',
    stacked=True,
    ax=ax2,
    color=[
        emotion_colors.get(col, '#CCCCCC')
        for col in emotion_cross.columns
    ],
    edgecolor='black'
)

ax2.set_title('Emotional Responses Across Violation Types', fontweight='bold', fontsize=13, pad=15)
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=30, pad=8)
ax2.legend(title='Primary Emotion', bbox_to_anchor=(1.02, 1), loc='upper left')


# CHART 3
ax3 = fig.add_subplot(gs[0, 2])

category_sentiment = (
    df.groupby('Trust_Category')['Sentiment']
    .agg(['mean', 'std'])
    .reindex(category_order)
)

x_pos = np.arange(len(category_sentiment))

ax3.bar(
    x_pos,
    category_sentiment['mean'],
    yerr=category_sentiment['std'],
    color=[
        colors_dict[cat]
        for cat in category_sentiment.index
    ],
    edgecolor='black',
    capsize=5
)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(category_sentiment.index, rotation=30, ha='right')
ax3.set_title('Average Sentiment by Violation Type', fontweight='bold', fontsize=13, pad=15)

ax3.set_ylabel('Sentiment Score (-1 to +1)')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)


# CHART 4
ax4 = fig.add_subplot(gs[1, 0])

negative_df = df[df['Trust_Category'] != 'Neutral/Positive']

if len(negative_df) > 0:
    emotion_counts = (negative_df['Primary_Emotion'].value_counts())
    ax4.pie(
        emotion_counts.values,
        labels=emotion_counts.index,
        autopct='%1.1f%%',
        colors=[
            emotion_colors.get(emotion, '#CCCCCC')
            for emotion in emotion_counts.index
        ],
        startangle=90, pctdistance=0.8, labeldistance=1.08
    )

    ax4.set_title('Emotional Fingerprint of Trust Violations', fontweight='bold', fontsize=13, pad=15)


# CHART 5
ax5 = fig.add_subplot(gs[1, 1])

integrity_reviews = ' '.join(df[df['Trust_Category'] == 'Integrity Violation']['review'])
competence_reviews = ' '.join(df[df['Trust_Category'] == 'Competence Violation']['review'])
integrity_words = ['hidden', 'unauthorized', 'lied', 'stole', 'scam', 'fraud']
competence_words = ['broken', 'slow', 'error', 'failed', 'bug', 'crash']
integrity_counts = {word: integrity_reviews.lower().count(word) for word in integrity_words}
competence_counts = {word: competence_reviews.lower().count(word) for word in competence_words}

x = np.arange(len(integrity_words))
width = 0.35

ax5.bar(x - width/2, list(integrity_counts.values()), width, label='Integrity', color=colors_dict['Integrity Violation'], edgecolor='black')
ax5.bar(x + width/2, list(competence_counts.values()), width, label='Competence', color=colors_dict['Competence Violation'], edgecolor='black')
ax5.set_xticks(x)
ax5.set_xticklabels(integrity_words, rotation=30, ha='right')
ax5.set_title('Key Trust Violation Trigger Words', fontweight='bold', fontsize=13, pad=15)
ax5.legend()

# CHART 6
ax6 = fig.add_subplot(gs[1, 2])
category_order = [
    'Integrity Violation',
    'Competence Violation',
    'General Dissatisfaction',
    'Neutral/Positive'
]

ordered_data = [
    df[df['Trust_Category'] == category]['Review_Length']
    for category in category_order
]

boxplot = ax6.boxplot(ordered_data, patch_artist=True, tick_labels=category_order)

for patch, category in zip(boxplot['boxes'], category_order):
    patch.set_facecolor(colors_dict[category])
    patch.set_alpha(0.80)

ax6.set_title('Review Length by Violation Type', fontweight='bold', fontsize=13, pad=15)
ax6.set_xlabel('')
ax6.set_ylabel('Character Count')
ax6.tick_params(axis='x', rotation=30, pad=8)

fig.suptitle('AI-Driven Brand Trust & Affective Sentiment Analyzer\nComprehensive Analysis Dashboard', fontsize=18, fontweight='bold',y=0.97)

plt.savefig(r'C:\Users\piyus\Documents\Codes\py\projects\AI-Driven Sentiment & Brand Trust Violation Analysis\trust_analysis_dashboard.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

plt.show()