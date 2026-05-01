import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pywaffle import Waffle
from train_model import X_test, X_train

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="NIDS", layout="wide")

# ------------------ INIT ------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None

df = st.session_state['df']
# Export for Tableau


# ------------------ SIDEBAR ------------------
with st.sidebar:

    st.markdown("## 🔐 NIDS")

    # Dark mode toggle
    dark_mode = st.toggle("🌙 Dark Mode")
    # Apply Dark / Light Theme
    if dark_mode:
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117;
                color: white;
            }
            .stSidebar {
                background-color: #161B22;
            }
            h1, h2, h3, h4, h5, h6, p, span {
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio("📂 Navigation", [
        "Dashboard",
        "EDA",
        "Model Results"
    ])

    st.markdown("---")

    # System Status
    st.markdown("### ⚙️ System Status")

    if df is not None:
        st.success("Data Loaded")
        st.success("Model Ready")
    else:
        st.warning("No Data Loaded")

    st.markdown("---")
    st.markdown("🔐 NIDS v1.0")
    st.caption("AI Intrusion Detection System")

# ------------------ DASHBOARD ------------------
if page == "Dashboard":

    st.markdown("# 🔐 AI-Based Network Intrusion Detection System")

    st.markdown("## 📂 Upload Dataset")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Store globally
        st.session_state['df'] = df

        st.success("✅ Dataset uploaded successfully!")

        st.markdown("### 📊 Dataset Preview")
        st.dataframe(df.head())

    else:
        st.info("⬆️ Please upload a dataset to continue")

# ------------------ LOAD DATA ------------------
df = st.session_state['df']

# ------------------ EDA ------------------
# ------------------ EDA ------------------
if page == "EDA":

    if df is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    st.markdown("## 📊 Exploratory Data Analysis")

    # ------------------ 1. ATTACK DISTRIBUTION ------------------
    # ------------------ 1. ATTACK DISTRIBUTION ------------------
    st.markdown("### 📌 Attack Distribution")

    import numpy as np

    if 'attack_cat' in df.columns:

    # Step 1: Count attacks
        attack_counts = df['attack_cat'].value_counts()

    # Step 2: Top 5 + Others
        top5 = attack_counts.head(5)
        others = attack_counts[5:].sum()

        attack_grouped = top5.copy()
        if others > 0:
            attack_grouped['Others'] = others

    # Step 3: Percentage
        attack_percent = (attack_grouped / attack_grouped.sum()) * 100

    # Step 4: Plot
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(6,6))

            wedges, _ = ax.pie(
                attack_percent,
                startangle=140
            )

        # Labels with arrows
            for i, wedge in enumerate(wedges):
                ang = (wedge.theta2 - wedge.theta1)/2 + wedge.theta1
                x = np.cos(np.deg2rad(ang))
                y = np.sin(np.deg2rad(ang))

                ax.annotate(
                    f"{attack_percent.index[i]}\n{attack_percent.values[i]:.1f}%",
                    xy=(x, y),
                    xytext=(1.4*np.sign(x), 1.4*y),
                    arrowprops=dict(arrowstyle="-"),
                    ha='center',
                    fontsize=9
                )

            ax.set_title("Top 5 Attack Categories + Others")

            st.pyplot(fig)
    # ------------------ 2. HISTOGRAM ------------------
    st.markdown("### 📊 Feature Histogram")

    num_cols = df.select_dtypes(include='number').columns

    if len(num_cols) > 0:

        col1, col2, col3 = st.columns([1.5,1,1.5])
        with col2:
            selected_col = st.selectbox("Select Feature", num_cols, key="hist")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.hist(df[selected_col], bins=25)

            ax.set_title(selected_col, fontsize=10)
            ax.tick_params(labelsize=8)

            plt.tight_layout()
            st.pyplot(fig)

    # ------------------ 3. CORRELATION HEATMAP ------------------
    st.markdown("### 🔥 Correlation Heatmap (Top Features)")

    # select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:

        # pick top 8 important features (variance based)
        top_features = numeric_df.var().sort_values(ascending=False).head(8).index

        corr = numeric_df[top_features].corr()

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            cax = ax.matshow(corr, cmap='coolwarm')

            fig.colorbar(cax)

            ax.set_xticks(range(len(top_features)))
            ax.set_yticks(range(len(top_features)))

            ax.set_xticklabels(top_features, rotation=45, fontsize=8)
            ax.set_yticklabels(top_features, fontsize=8)

            st.pyplot(fig)

    # ------------------ 4. BOXPLOT ------------------
    st.markdown("### 📦 Boxplot")

    if len(num_cols) > 0:

        col1, col2, col3 = st.columns([1.5,1,1.5])
        with col2:
            selected_box = st.selectbox("Select Feature for Boxplot", num_cols, key="box")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.boxplot(df[selected_box])

            ax.set_title(selected_box, fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            
            
        # ------------------ 5. SCATTER PLOT ------------------
    st.markdown("### 🔵 Scatter Plot (Traffic Relationship)")

    if 'sbytes' in df.columns and 'dbytes' in df.columns:

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(5,4))

        # Color by attack category if available
            if 'attack_cat' in df.columns:
                categories = df['attack_cat'].astype('category').cat.codes
                scatter = ax.scatter(df['sbytes'], df['dbytes'], c=categories)
            else:
                scatter = ax.scatter(df['sbytes'], df['dbytes'])

            ax.set_xlabel("sbytes")
            ax.set_ylabel("dbytes")
            ax.set_title("sbytes vs dbytes")

            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.warning("Required columns (sbytes, dbytes) not found")
            
        # ------------------ 6. VIOLIN PLOT ------------------
    st.markdown("###  Violin Plot (Distribution by Attack)")

    if 'attack_cat' in df.columns and 'sbytes' in df.columns:

        import seaborn as sns

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))

            sns.violinplot(
                x=df['attack_cat'],
                y=df['sbytes'],
                ax=ax
            )

            ax.set_title("Traffic Distribution by Attack Type")
            ax.set_xlabel("Attack Category")
            ax.set_ylabel("sbytes")

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

    else:
        st.warning("Required columns not found")
        
        
        from pywaffle import Waffle

    st.markdown("### 🧱 Waffle Chart (Top Attacks)")

    if 'attack_cat' in df.columns:

        attack_counts = df['attack_cat'].value_counts()

    # Top 5 + Others
        top5 = attack_counts.head(5)
        others = attack_counts[5:].sum()

        attack_grouped = top5.copy()
        if others > 0:
            attack_grouped['Others'] = others

        fig = plt.figure(
            FigureClass=Waffle,
            rows=5,
            columns=20,
            values=attack_grouped.to_dict(),
            figsize=(10,4),
            title={'label': 'Top 5 Attacks + Others', 'loc': 'center'},
            legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)}
        )
        

        st.pyplot(fig)

# ------------------ MODEL RESULTS ------------------
    # ------------------ MODEL RESULTS ------------------
if page == "Model Results":

    if df is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    st.markdown("## 🤖 Model Results")

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    import numpy as np
    
    # ------------------ MODEL TRAINING (CACHED) ------------------



    
    

    # ------------------ TARGET ------------------
    if 'attack_cat' in df.columns:
        y = df['attack_cat']
        X = df.drop(columns=['attack_cat'])
    elif 'label' in df.columns:
        y = df['label']
        X = df.drop(columns=['label'])
    else:
        st.error("❌ No target column found")
        st.stop()

    # ------------------ ENCODE FEATURES ------------------
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # ------------------ ENCODE TARGET ------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ------------------ SCALE ------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------ SPLIT ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    # ------------------ MODEL COMPARISON (ADD THIS) ------------------
    # ------------------ MODEL COMPARISON ------------------

    st.markdown("### 📊 Model Accuracy Comparison")

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_temp = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_temp)
        results[name] = acc

# Convert to DataFrame
    results_df = pd.DataFrame(
        list(results.items()),
        columns=["Model", "Accuracy"]
    )

    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    st.dataframe(results_df)

# Bar chart
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(results_df["Model"], results_df["Accuracy"])
    plt.xticks(rotation=45)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

# Best model
    best_model_name = results_df.iloc[0]["Model"]
    best_accuracy = results_df.iloc[0]["Accuracy"]

    st.success(f"🏆 Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")

    # ------------------ MODEL ------------------
    # ------------------ USE BEST MODEL ------------------
    model = models[best_model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Step 1: Copy original dataset (keeps all columns)
    df_export = df.copy()

# Step 2: Generate predictions (ONLY once)
    y_pred = model.predict(X_scaled)

# Step 3: Convert predictions back to original labels
    df_export['AI Predicted Label'] = label_encoder.inverse_transform(y_pred)

# Step 4: Convert actual labels back (IMPORTANT FIX)
    df_export['Actual Label'] = y

# Step 5: Check correctness
    df_export['Is AI Correct'] = df_export['Actual Label'] == df_export['AI Predicted Label']

# Step 6: (Optional – keep if needed for Tableau compatibility)
    df_export['predicted_attack'] = df_export['AI Predicted Label']

# Step 7: Save CSV
    df_export.to_csv("nids_tableau_export.csv", index=False)

# Step 8: Confirmation
    st.success("📁 Data exported for Tableau with correct labels")

    # ------------------ ACCURACY ------------------

    accuracy = model.score(X_test, y_test)

    st.success(f"Model Accuracy: {accuracy*100:.2f}%")
    
    
    # ------------------ FEATURE IMPORTANCE ------------------
    st.markdown("### 🔍 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_
        feature_names = X.columns

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        feat_df = feat_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))

            ax.barh(
                feat_df["Feature"],
                feat_df["Importance"]
            )

            ax.invert_yaxis()
            ax.set_title("Top 10 Important Features")

            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Feature importance not available for this model")

    # ------------------ CONFUSION MATRIX ------------------

    st.subheader("📊 Confusion Matrix")

    import seaborn as sns
    y_pred=model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            cm,
            cmap="Greens",
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title("Confusion Matrix", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)

    # ------------------ CLASSIFICATION REPORT ------------------
    st.markdown("### 📄 Classification Report")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(report)

    
    # ------------------ MULTI-CLASS ROC CURVE ------------------
    st.markdown("### 📈 ROC Curve (Multi-Class)")

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

# Convert labels to binary format
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

# Get probability scores
    y_score = model.predict_proba(X_test)

# Plot
    col1, col2, col3 = st.columns([1,2,1])
    with col2:

        fig, ax = plt.subplots(figsize=(4,3))

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = label_encoder.inverse_transform([classes[i]])[0]

            ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

    # Diagonal line
        ax.plot([0,1], [0,1], 'k--')

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Multi-Class ROC Curve")

        ax.legend(fontsize=6)

        st.pyplot(fig)

    # ------------------ SINGLE ROW PREDICTION ------------------
        st.markdown("### 🔍 Single Row Prediction")

        row_num = st.number_input(
            "Enter Row Number",
            min_value=0,
            max_value=len(df)-1,
            step=1
        )

        if st.button("Predict"):

        # ------------------ SHOW ORIGINAL ROW ------------------
            original_row = df.iloc[[row_num]]

            st.markdown("### 📄 Selected Row Data")
            st.dataframe(original_row)

        # ------------------ PREPARE INPUT ------------------
            input_data = X.iloc[[row_num]]
            input_scaled = scaler.transform(input_data)

        # ------------------ PREDICT ------------------
            prediction = model.predict(input_scaled)
            pred_label = label_encoder.inverse_transform(prediction)[0]

        # ------------------ CONFIDENCE ------------------
            prob = model.predict_proba(input_scaled)
            confidence = np.max(prob)

        # ------------------ DISPLAY RESULT ------------------
            st.markdown("### 🎯 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Attack Type", pred_label)

            with col2:
                st.metric("Confidence", f"{confidence*100:.2f}%")

        # ------------------ FINAL STATUS ------------------
            if str(pred_label).lower() == "normal":
                st.success("✅ Normal Traffic")
            else:
                st.error(f"🚨 Attack Detected: {pred_label}")