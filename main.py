import io
import sqlite3
import bcrypt
from pydantic import BaseModel


from pathlib import Path
import json
import io

import torch
import joblib
import pandas as pd
import numpy as np

from pynndescent import NNDescent
from torch_geometric.utils import to_undirected
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch import nn
from torch_geometric.nn import SAGEConv


# ==========================================
# PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device("cpu")
INPUT_FEATURES = 46

K_NEIGHBORS = 5
BATCH_SIZE = 20000


# ==========================================
# GRAPHSAGE MODEL ARCHITECTURE
# ==========================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1,
        hidden2,
        num_classes,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden1)
        self.conv2 = SAGEConv(hidden1, hidden2)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        return self.classifier(x)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="ThreatSense API",
    description="Multistage GraphSAGE Intrusion Detection System",
    version="1.0.0"
)
# ==========================================
# THREATSENSE USER DATABASE
# ==========================================

DATABASE_PATH = BASE_DIR / "threatsense_users.db"


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection





def create_user_table():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def create_history_tables():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS activity_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER,
            activity_type TEXT NOT NULL,
            filename TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (session_id) REFERENCES user_sessions(id)
        )
    """)

    connection.commit()
    connection.close()


create_user_table()
create_history_tables()

# ==========================================
# AUTHENTICATION REQUEST MODELS
# ==========================================

class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# GLOBAL MODEL STORAGE
# ==========================================

stage_a_model = None
stage_b_model = None

feature_scaler = None
stage_b_scaler = None

feature_columns = None
label_mapping = None

MODEL_STATUS = {
    "stage_a": "not loaded",
    "stage_b": "not loaded",
    "feature_scaler": "not loaded",
    "stage_b_scaler": "not loaded",
    "feature_columns": "not loaded",
    "label_mapping": "not loaded"
}
# Stores the latest graph for frontend visualization
latest_topology = {
    "nodes": [],
    "links": []
}


# ==========================================
# GRAPH CONSTRUCTION
# ==========================================

def build_graph(X):

    if len(X) <= K_NEIGHBORS:
        raise ValueError(
            f"At least {K_NEIGHBORS + 1} records "
            "are required to build a graph."
        )

    nn = NNDescent(
        X,
        n_neighbors=K_NEIGHBORS + 1,
        metric="euclidean",
        random_state=42,
        n_jobs=2
    )

    neighbours, _ = nn.neighbor_graph

    source = np.repeat(
        np.arange(len(X)),
        K_NEIGHBORS
    )

    target = neighbours[:, 1:K_NEIGHBORS + 1].reshape(-1)

    edge_index = np.vstack([
        np.concatenate([source, target]),
        np.concatenate([target, source])
    ])

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    )

    return to_undirected(
        edge_index,
        num_nodes=len(X)
    )


# ==========================================
# FEATURE PREPROCESSING
# ==========================================

def prepare_features(df):

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_scaled = feature_scaler.transform(X)

    return X_scaled.astype(np.float32)


# ==========================================
# LOAD STAGE A MODEL
# ==========================================

def load_stage_a():

    global stage_a_model

    path = MODEL_DIR / "stage_a_graphsage.pt"

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    stage_a_model = GraphSAGE(
        input_dim=checkpoint["input_dim"],
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"]
    )

    stage_a_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    stage_a_model.to(DEVICE)
    stage_a_model.eval()

    return {
        "status": "loaded",
        "input_features": checkpoint["input_dim"],
        "classes": checkpoint["num_classes"],
        "validation_macro_f1": checkpoint.get(
            "best_validation_macro_f1"
        )
    }


# ==========================================
# LOAD STAGE B MODEL
# ==========================================

def load_stage_b():

    global stage_b_model

    path = MODEL_DIR / "stage_b_graphsage.pt"

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    stage_b_model = GraphSAGE(
        input_dim=46,
        hidden1=256,
        hidden2=128,
        num_classes=3,
        dropout=0.2
    )

    stage_b_model.load_state_dict(
        checkpoint["model_state"]
    )

    stage_b_model.to(DEVICE)
    stage_b_model.eval()

    return {
        "status": "loaded",
        "input_features": 46,
        "classes": 3,
        "best_validation_f1": checkpoint.get("best_val_f1"),
        "validation_accuracy": checkpoint.get(
            "validation_accuracy"
        ),
        "training_epoch": checkpoint.get("epoch")
    }


# ==========================================
# LOAD PREPROCESSING FILES
# ==========================================

def load_preprocessing():

    global feature_scaler
    global stage_b_scaler
    global feature_columns
    global label_mapping

    feature_scaler = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

    stage_b_scaler = joblib.load(
        MODEL_DIR / "stage_b_scaler.joblib"
    )

    with open(
        MODEL_DIR / "feature_columns.json",
        "r",
        encoding="utf-8"
    ) as f:
        feature_columns = json.load(f)

    with open(
        MODEL_DIR / "label_mapping.json",
        "r",
        encoding="utf-8"
    ) as f:
        label_mapping = json.load(f)

    if len(feature_columns) != INPUT_FEATURES:
        raise ValueError(
            f"Expected 46 features, found {len(feature_columns)}"
        )

    if feature_scaler.n_features_in_ != INPUT_FEATURES:
        raise ValueError("Feature scaler mismatch")

    if stage_b_scaler.n_features_in_ != INPUT_FEATURES:
        raise ValueError("Stage B scaler mismatch")

    return {
        "feature_count": len(feature_columns),
        "feature_scaler": "loaded",
        "stage_b_scaler": "loaded",
        "label_mapping": "loaded"
    }


# ==========================================
# MODEL INITIALIZATION
# ==========================================

@app.on_event("startup")
def initialize_models():

    try:
        MODEL_STATUS["stage_a"] = load_stage_a()

    except Exception as e:
        MODEL_STATUS["stage_a"] = f"ERROR: {str(e)}"

    try:
        MODEL_STATUS["stage_b"] = load_stage_b()

    except Exception as e:
        MODEL_STATUS["stage_b"] = f"ERROR: {str(e)}"

    try:
        load_preprocessing()

        MODEL_STATUS["feature_scaler"] = "loaded"
        MODEL_STATUS["stage_b_scaler"] = "loaded"
        MODEL_STATUS["feature_columns"] = "loaded"
        MODEL_STATUS["label_mapping"] = "loaded"

    except Exception as e:
        error = f"ERROR: {str(e)}"

        for key in [
            "feature_scaler",
            "stage_b_scaler",
            "feature_columns",
            "label_mapping"
        ]:
            MODEL_STATUS[key] = error


# ==========================================
# STAGE A PREDICTION
# ==========================================

def predict_stage_a(X):
    global latest_topology

    if stage_a_model is None:
        raise RuntimeError("Stage A model is not loaded.")

    # Build the actual KNN graph
    edge_index = build_graph(X)

    x_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    edge_index = edge_index.to(DEVICE)

    with torch.no_grad():
        logits = stage_a_model(
            x_tensor,
            edge_index
        )
        predictions = logits.argmax(dim=1)

    predictions = predictions.cpu().numpy()

    # Prepare a manageable visualization sample
    max_nodes = min(len(X), 300)

    nodes = [
        {
            "id": i,
            "label": f"Record {i + 1}",
            "type": (
                "malicious"
                if int(predictions[i]) == 1
                else "benign"
            )
        }
        for i in range(max_nodes)
    ]

    # Keep only actual KNN edges between displayed nodes
    edges = edge_index.cpu().numpy()

    links = [
        {
            "source": int(source),
            "target": int(target)
        }
        for source, target in edges.T
        if source < max_nodes and target < max_nodes
    ]

    latest_topology = {
        "nodes": nodes,
        "links": links,
        "total_graph_nodes": len(X),
        "displayed_nodes": max_nodes,
        "total_graph_edges": int(edge_index.size(1) // 2)
    }

    return predictions



# ==========================================
# STAGE B PREDICTION
# ==========================================

def predict_stage_b(X):

    if stage_b_model is None:
        raise RuntimeError("Stage B model is not loaded.")

    if len(X) <= K_NEIGHBORS:
        return None

    edge_index = build_graph(X)

    x_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    edge_index = edge_index.to(DEVICE)

    with torch.no_grad():

        logits = stage_b_model(
            x_tensor,
            edge_index
        )

        predictions = logits.argmax(dim=1)

    return predictions.cpu().numpy()


# ==========================================
# LABEL CONVERSION
# ==========================================

def get_stage_a_label(prediction):

    return "malicious" if int(prediction) == 1 else "benign"


def get_stage_b_label(prediction):

    labels = {
        0: "LOW",
        1: "MILD",
        2: "HIGH"
    }

    return labels.get(int(prediction), "UNKNOWN")


# ==========================================
# API ROUTES
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to ThreatSense!",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "device": str(DEVICE)
    }


@app.get("/models/status")
def models_status():

    return {
        "project": "ThreatSense",
        "models": MODEL_STATUS
    }
@app.get("/topology")
def get_topology():
    return latest_topology

# ==========================================
# CSV UPLOAD AND PREDICTION API
# ==========================================

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        if (
            stage_a_model is None
            or stage_b_model is None
            or feature_scaler is None
            or feature_columns is None
        ):
            raise HTTPException(
                status_code=503,
                detail="Models or preprocessing files are not loaded."
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no records."
            )

        missing_columns = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "CSV is missing required model features.",
                    "missing_columns": missing_columns
                }
            )

        all_predictions = []

        for start in range(0, len(df), BATCH_SIZE):

            batch = df.iloc[
                start:start + BATCH_SIZE
            ].copy()

            X_scaled = prepare_features(batch)

            if len(X_scaled) <= K_NEIGHBORS:
                raise ValueError(
                    "Each processing batch must contain at least 6 records."
                )

            stage_a_predictions = predict_stage_a(
                X_scaled
            )

            stage_b_predictions = np.full(
                len(batch),
                -1,
                dtype=int
            )

            malicious_indices = np.where(
                stage_a_predictions == 1
            )[0]

            if len(malicious_indices) > K_NEIGHBORS:

                malicious_features = X_scaled[
                    malicious_indices
                ]

                severity_predictions = predict_stage_b(
                    malicious_features
                )

                if severity_predictions is not None:

                    stage_b_predictions[
                        malicious_indices
                    ] = severity_predictions

            for i in range(len(batch)):

                stage_a_label = get_stage_a_label(
                    stage_a_predictions[i]
                )

                if stage_a_predictions[i] == 0:

                    severity = "NOT_APPLICABLE"
                    final_label = "BENIGN"

                elif stage_b_predictions[i] == -1:

                    severity = "INSUFFICIENT_RECORDS"
                    final_label = "MALICIOUS_UNCLASSIFIED"

                else:

                    severity = get_stage_b_label(
                        stage_b_predictions[i]
                    )

                    final_label = severity

                all_predictions.append({
                    "record": start + i + 1,
                    "stage_a_prediction": stage_a_label,
                    "stage_b_severity": severity,
                    "final_prediction": final_label
                })

        stage_a_counts = {
            "benign": sum(
                p["stage_a_prediction"] == "benign"
                for p in all_predictions
            ),
            "malicious": sum(
                p["stage_a_prediction"] == "malicious"
                for p in all_predictions
            )
        }

        severity_counts = {
            label: sum(
                p["stage_b_severity"] == label
                for p in all_predictions
            )
            for label in ["LOW", "MILD", "HIGH"]
        }

        return {
            "message": "CSV processed successfully.",
            "filename": file.filename,
            "total_records": len(df),
            "total_columns": len(df.columns),
            "original_rows": json.loads(df.to_json(orient="records")),
            "stage_a_results": stage_a_counts,
            "stage_b_results": severity_counts,
            "predictions": all_predictions,
            "returned_prediction_records": min(
                len(all_predictions), 100
            ),
            "note": (
                "Only the first 100 individual predictions are returned. "
                "Stage B classification requires more than 5 malicious "
                "records in a processing batch."
            ),
            "status": "analysis_completed"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        await file.close()
        
@app.post("/auth/register")
def register_user(user: RegisterRequest):
    if len(user.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters."
        )

    password_hash = bcrypt.hashpw(
        user.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    connection = get_db_connection()

    try:
        connection.execute(
            """
            INSERT INTO users (full_name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (user.full_name, user.email.lower(), password_hash)
        )
        connection.commit()

        return {"message": "Registration successful."}

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    finally:
        connection.close()
@app.post("/auth/login")
def login_user(user: LoginRequest):
    connection = get_db_connection()

    try:
        saved_user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (user.email.lower(),)
        ).fetchone()

        if saved_user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        password_matches = bcrypt.checkpw(
            user.password.encode("utf-8"),
            saved_user["password_hash"].encode("utf-8")
        )

        if not password_matches:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        connection.execute(
            "INSERT INTO user_sessions (user_id) VALUES (?)",
            (saved_user["id"],)
        )
        connection.commit()

        session_id = connection.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]

        return {
            "message": "Login successful.",
            "session_id": session_id,
            "user": {
                "id": saved_user["id"],
                "full_name": saved_user["full_name"],
                "email": saved_user["email"]
            }
        }

    finally:
        connection.close()

@app.post("/auth/logout/{session_id}")
def logout_user(session_id: int):

    connection = get_db_connection()

    try:
        cursor = connection.execute(
            """
            UPDATE user_sessions
            SET logout_at = CURRENT_TIMESTAMP
            WHERE id = ? AND logout_at IS NULL
            """,
            (session_id,)
        )

        connection.commit()

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Active session not found."
            )

        return {"message": "Logout successful."}
    finally:
        connection.close()


class ActivityRequest(BaseModel):
    user_id: int
    session_id: int
    activity_type: str
    filename: str = ""
    details: str = ""


@app.post("/history/activity")
def save_activity(activity: ActivityRequest):
    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO activity_history
            (user_id, session_id, activity_type, filename, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                activity.user_id,
                activity.session_id,
                activity.activity_type,
                activity.filename,
                activity.details,
            ),
        )
        connection.commit()
        return {"message": "Activity saved."}
    finally:
        connection.close()


@app.get("/history/{user_id}")
def get_history(user_id: int):
    connection = get_db_connection()
    try:
        sessions = connection.execute(
            """
            SELECT id, session_id, activity_type, filename, details, created_at
FROM activity_history
WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

        activities = connection.execute(
            """
            SELECT id, activity_type, filename, details, created_at
            FROM activity_history
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

        history = []

        for session in sessions:
            history.append({
                "id": f"login-{session['id']}",
                "activity_type": "Login",
                "filename": "",
                "details": "User logged in",
                "created_at": session["login_at"],
            })

            if session["logout_at"]:
                history.append({
                    "id": f"logout-{session['id']}",
                    "activity_type": "Logout",
                    "filename": "",
                    "details": "User logged out",
                    "created_at": session["logout_at"],
                })

        for activity in activities:
            history.append(dict(activity))

        history.sort(
            key=lambda item: item["created_at"] or "",
            reverse=True,
        )

        return {"history": history}
    finally:
        connection.close()