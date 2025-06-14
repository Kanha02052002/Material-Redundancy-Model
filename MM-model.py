"""
DOCUMENTATION -
1. Fuzzywuzzy       -> https://pypi.org/project/fuzzywuzzy/
2. Pandas           -> https://pandas.pydata.org/docs/
3. Numpy            -> https://numpy.org/doc/stable/index.html
4. Dataclasses      -> https://docs.python.org/3/library/dataclasses.html
5. Matplotlib       -> https://matplotlib.org/stable/index.html
6. Seaborn          -> https://seaborn.pydata.org/
7. Plotly           -> https://plotly.com/python/
8. Scikit-Learn     -> https://scikit-learn.org/stable/
9. Torch            -> https://docs.pytorch.org/docs/stable/index.html
10. Torch-Geometirc -> https://pytorch-geometric.readthedocs.io/en/latest/  [for Graph Neural Network]
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple, Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T

try:
    from sentence_transformers import SentenceTransformer
    SLM_AVAILABLE = True
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers", "torch-geometric"])
    from sentence_transformers import SentenceTransformer
    SLM_AVAILABLE = True

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz, process
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class MultiApproachConfig:
    """Configuration for multi-approach material grouping."""
    
    # SLM Configuration
    slm_model_name: str = "all-mpnet-base-v2"   # "all-MiniLM-L6-v2"
    use_slm: bool = True
    slm_batch_size: int = 32
    
    # GNN Configuration
    use_gnn: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 5
    gnn_dropout: float = 0.2
    
    # Hybrid Learning Configuration
    use_ensemble: bool = True
    ensemble_methods: List[str] = None
    
    # Similarity thresholds
    fuzzy_threshold: float = 80.0
    cosine_threshold: float = 0.65
    slm_threshold: float = 0.75
    gnn_threshold: float = 0.70
    ensemble_threshold: float = 0.75
    
    # Clustering parameters
    eps: float = 0.25
    min_samples: int = 2
    
    # Text processing
    max_features: int = 3000
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Weights for ensemble
    slm_weight: float = 0.3
    gnn_weight: float = 0.3
    tfidf_weight: float = 0.2
    fuzzy_weight: float = 0.2
    
    # Advanced parameters
    use_adaptive_threshold: bool = True
    use_outlier_detection: bool = True
    use_hierarchical: bool = True
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['dbscan', 'hierarchical', 'spectral', 'kmeans']

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for material similarity learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize GNN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output embedding dimension
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, concat=False, dropout=dropout))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            batch (torch.Tensor): Batch indices
            
        Returns:
            torch.Tensor: Node embeddings
        """
        # Graph convolution layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final projection
        x = self.projection(x)
        
        return x

class EnhancedTextPreprocessor:
    """Enhanced text preprocessing with domain-specific optimizations."""
    
    def __init__(self):
        """Initialize the enhanced text preprocessor."""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Extended technical stopwords
        self.technical_stopwords = {
            'as', 'per', 'spec', 'rev', 'drg', 'no', 'item', 'qty', 'nos',
            'mm', 'kg', 'mt', 'mr', 'lt', 'kl', 'st', 'sh', 'lo', 'qm',
            'dia', 'thk', 'lg', 'od', 'id', 'var', 'rev', 'bhel', 'make',
            'type', 'grade', 'class', 'size', 'length', 'width', 'thick'
        }
        
        self.stop_words.update(self.technical_stopwords)
        
        # Material type patterns    ->  can be changed later on
        """Material spelling grouping according to similar format. Can be add more later on."""
        self.material_patterns = {
            'steel': r'\b(steel|stl|ss|ms|cs)\b',
            'copper': r'\b(copper|cu|brass)\b',
            'aluminum': r'\b(aluminum|aluminium|al)\b',
            'pipe': r'\b(pipe|tube|hose)\b',
            'bolt': r'\b(bolt|screw|stud|nut)\b',
            'valve': r'\b(valve|gate|globe|check)\b',
            'bearing': r'\b(bearing|brg|bush)\b',
            'transformer': r'\b(transformer|trf|ct|pt)\b'
        }
    
    def extract_material_type(self, text: str) -> str:
        """Extract material type from description."""
        text_lower = text.lower()
        for material_type, pattern in self.material_patterns.items():
            if re.search(pattern, text_lower):
                return material_type
        return 'general'
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Normalize abbreviations -> Add more later on
        abbreviations = {
            r'\bdia\b': 'diameter',
            r'\bthk\b': 'thick',
            r'\blg\b': 'length',
            r'\bwt\b': 'weight',
            r'\bqty\b': 'quantity'
        }
        
        for abbrev, full_form in abbreviations.items():
            text = re.sub(abbrev, full_form, text)
        
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d+(?:\.\d+)?\b', '', text)
        
        return text.strip()
    
    def extract_enhanced_keywords(self, text: str) -> str:
        """Extract enhanced keywords with material context."""
        cleaned_text = self.clean_text(text)
        material_type = self.extract_material_type(text)
        
        words = [word for word in cleaned_text.split() 
                if word not in self.stop_words and len(word) > 2]
        
        stemmed_words = [self.stemmer.stem(word) for word in words]
        
        if material_type != 'general':
            stemmed_words.append(material_type)
        
        return ' '.join(stemmed_words)

class MultiApproachSimilarityCalculator:
    """Multi-approach similarity calculator combining SLM, GNN, and traditional methods."""
    
    def __init__(self, config: MultiApproachConfig):
        """Initialize multi-approach similarity calculator."""
        self.config = config
        
        # Traditional methods
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # SLM processor
        self.slm_model = None
        self.slm_embeddings = None
        
        # GNN components
        self.gnn_model = None
        self.graph_data = None
        self.gnn_embeddings = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if config.use_slm:
            self._initialize_slm()
    
    def _initialize_slm(self):
        """Initialize SLM model."""
        try:
            self.logger.info(f"Loading SLM model: {self.config.slm_model_name}")
            self.slm_model = SentenceTransformer(self.config.slm_model_name)
            self.logger.info("SLM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SLM model: {e}")
            self.config.use_slm = False
    
    def fit_all_models(self, texts: List[str], numerical_features: np.ndarray):
        """Fit all similarity models."""
        self.logger.info("Fitting all similarity models...")
        
        # Fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Generate SLM embeddings
        if self.config.use_slm and self.slm_model:
            self.logger.info("Generating SLM embeddings...")
            self.slm_embeddings = self.slm_model.encode(
                texts, 
                batch_size=self.config.slm_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        # Build and train GNN
        if self.config.use_gnn:
            self._build_and_train_gnn(texts, numerical_features)
    
    def _build_and_train_gnn(self, texts: List[str], numerical_features: np.ndarray):
        """Build and train GNN model."""
        self.logger.info("Building and training GNN...")
        
        # Create node features (combine text embeddings and numerical features)
        if self.slm_embeddings is not None:
            text_features = self.slm_embeddings
        else:
            # Fallback to TF-IDF if SLM not available
            text_features = self.tfidf_matrix.toarray()
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)
        
        # Combine features
        node_features = np.concatenate([text_features, numerical_features_scaled], axis=1)
        
        # Build graph edges based on similarity
        edge_indices = self._build_graph_edges(text_features)
        
        # Create PyTorch Geometric data
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        self.graph_data = Data(x=x, edge_index=edge_index)
        
        # Initialize and train GNN
        input_dim = node_features.shape[1]
        self.gnn_model = GraphNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.gnn_hidden_dim,
            output_dim=self.config.gnn_hidden_dim,
            num_layers=self.config.gnn_num_layers,
            dropout=self.config.gnn_dropout
        )
        
        # Train GNN
        self._train_gnn()
    
    def _build_graph_edges(self, features: np.ndarray, threshold: float = 0.3) -> List[List[int]]:
        """Build graph edges based on feature similarity."""
        similarity_matrix = cosine_similarity(features)
        edges = []
        
        n_nodes = len(features)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if similarity_matrix[i, j] > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
        
        return edges
    
    def _train_gnn(self, epochs: int = 100):
        """Train GNN model using contrastive learning."""
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.001)
        
        self.gnn_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)
            
            # Contrastive loss
            loss = self._compute_contrastive_loss(embeddings)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.info(f"GNN Training Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Generate final embeddings
        self.gnn_model.eval()
        with torch.no_grad():
            self.gnn_embeddings = self.gnn_model(
                self.graph_data.x, self.graph_data.edge_index
            ).numpy()
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, temperature: float = 0.1):
        """Compute contrastive loss for GNN training."""
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size, device=embeddings.device)
        
        return F.cross_entropy(similarity_matrix, labels)
    
    def calculate_multi_approach_similarity(self, idx1: int, idx2: int, 
                                          text1: str, text2: str,
                                          unit1: str, unit2: str) -> float:
        """Calculate similarity using multiple approaches."""
        similarities = {}
        
        # Fuzzy similarity
        similarities['fuzzy'] = self._calculate_fuzzy_similarity(text1, text2)
        
        # TF-IDF similarity
        similarities['tfidf'] = self._calculate_tfidf_similarity(idx1, idx2)
        
        # SLM similarity
        if self.config.use_slm and self.slm_embeddings is not None:
            similarities['slm'] = self._calculate_slm_similarity(idx1, idx2)
        else:
            similarities['slm'] = 0.0
        
        # GNN similarity
        if self.config.use_gnn and self.gnn_embeddings is not None:
            similarities['gnn'] = self._calculate_gnn_similarity(idx1, idx2)
        else:
            similarities['gnn'] = 0.0
        
        # Unit similarity
        similarities['unit'] = self._calculate_unit_similarity(unit1, unit2)
        
        # Weighted ensemble
        ensemble_score = (
            self.config.fuzzy_weight * similarities['fuzzy'] +
            self.config.tfidf_weight * similarities['tfidf'] +
            self.config.slm_weight * similarities['slm'] +
            self.config.gnn_weight * similarities['gnn'] +
            0.1 * similarities['unit']  # Small weight for unit similarity
        )
        
        return ensemble_score
    
    def _calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy similarity."""
        return max(fuzz.token_sort_ratio(text1, text2), fuzz.partial_ratio(text1, text2)) / 100.0
    
    def _calculate_tfidf_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate TF-IDF similarity."""
        if self.tfidf_matrix is None:
            return 0.0
        
        vec1 = self.tfidf_matrix[idx1]
        vec2 = self.tfidf_matrix[idx2]
        return cosine_similarity(vec1, vec2)[0, 0]
    
    def _calculate_slm_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate SLM similarity."""
        if self.slm_embeddings is None:
            return 0.0
        
        vec1 = self.slm_embeddings[idx1].reshape(1, -1)
        vec2 = self.slm_embeddings[idx2].reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0, 0]
    
    def _calculate_gnn_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate GNN similarity."""
        if self.gnn_embeddings is None:
            return 0.0
        
        vec1 = self.gnn_embeddings[idx1].reshape(1, -1)
        vec2 = self.gnn_embeddings[idx2].reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0, 0]
    
    def _calculate_unit_similarity(self, unit1: str, unit2: str) -> float:
        """Calculate unit similarity."""
        if pd.isna(unit1) or pd.isna(unit2):
            return 0.0
        
        unit1_clean = unit1.strip().upper()
        unit2_clean = unit2.strip().upper()
        
        if unit1_clean == unit2_clean:
            return 1.0
        
        fuzzy_score = fuzz.ratio(unit1_clean, unit2_clean) / 100.0
        return fuzzy_score if fuzzy_score > 0.8 else 0.0

class EnsembleClusterer:
    """Ensemble clustering using multiple algorithms."""
    
    def __init__(self, config: MultiApproachConfig):
        """Initialize ensemble clusterer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit_predict_ensemble(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Perform ensemble clustering."""
        self.logger.info("Performing ensemble clustering...")
        
        distance_matrix = 1 - similarity_matrix
        clustering_results = {}
        
        # DBSCAN
        if 'dbscan' in self.config.ensemble_methods:
            dbscan = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples, 
                           metric='precomputed')
            clustering_results['dbscan'] = dbscan.fit_predict(distance_matrix)
        
        # Hierarchical Clustering
        if 'hierarchical' in self.config.ensemble_methods:
            n_clusters = max(2, len(np.unique(clustering_results.get('dbscan', [0]))) - 
                           (1 if -1 in clustering_results.get('dbscan', []) else 0))
            n_clusters = min(n_clusters, len(similarity_matrix) // 3)
            
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, 
                                                 metric='precomputed', linkage='average')
            clustering_results['hierarchical'] = hierarchical.fit_predict(distance_matrix)
        
        # Spectral Clustering
        if 'spectral' in self.config.ensemble_methods:
            try:
                n_clusters = max(2, len(np.unique(clustering_results.get('dbscan', [0]))) - 
                               (1 if -1 in clustering_results.get('dbscan', []) else 0))
                n_clusters = min(n_clusters, len(similarity_matrix) // 3)
                
                spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
                clustering_results['spectral'] = spectral.fit_predict(similarity_matrix)
            except Exception as e:
                self.logger.warning(f"Spectral clustering failed: {e}")
        
        # K-Means on embeddings
        if 'kmeans' in self.config.ensemble_methods:
            try:
                # Use PCA to reduce dimensionality for K-means
                pca = PCA(n_components=min(50, similarity_matrix.shape[0] - 1))
                reduced_features = pca.fit_transform(similarity_matrix)
                
                n_clusters = max(2, len(np.unique(clustering_results.get('dbscan', [0]))) - 
                               (1 if -1 in clustering_results.get('dbscan', []) else 0))
                n_clusters = min(n_clusters, len(similarity_matrix) // 3)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clustering_results['kmeans'] = kmeans.fit_predict(reduced_features)
            except Exception as e:
                self.logger.warning(f"K-means clustering failed: {e}")
        
        # Ensemble voting
        final_labels = self._ensemble_voting(clustering_results, similarity_matrix)
        
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        self.logger.info(f"Ensemble clustering completed: {n_clusters} clusters")
        
        return final_labels
    
    def _ensemble_voting(self, clustering_results: Dict[str, np.ndarray], 
                        similarity_matrix: np.ndarray) -> np.ndarray:
        """Perform ensemble voting on clustering results."""
        if not clustering_results:
            return np.zeros(len(similarity_matrix))
        
        # Use the first available result as base
        base_method = list(clustering_results.keys())[0]
        final_labels = clustering_results[base_method].copy()
        
        # For each method, try to improve the base clustering
        for method_name, labels in clustering_results.items():
            if method_name == base_method:
                continue
            
            # Consensus-based refinement
            final_labels = self._refine_clustering(final_labels, labels, similarity_matrix)
        
        return final_labels
    
    def _refine_clustering(self, base_labels: np.ndarray, new_labels: np.ndarray, 
                          similarity_matrix: np.ndarray) -> np.ndarray:
        """Refine clustering using consensus."""
        refined_labels = base_labels.copy()
        
        # Handle noise points in base clustering
        noise_indices = np.where(base_labels == -1)[0]
        
        for idx in noise_indices:
            # Find the cluster assignment from new_labels
            if new_labels[idx] != -1:
                # Check if this assignment makes sense based on similarity
                cluster_items = np.where(new_labels == new_labels[idx])[0]
                avg_similarity = np.mean([similarity_matrix[idx, j] for j in cluster_items if j != idx])
                
                if avg_similarity > self.config.ensemble_threshold:
                    refined_labels[idx] = new_labels[idx]
        
        return refined_labels

class MultiApproachMaterialGrouper:
    """Multi-approach material grouper combining SLM, GNN, and ensemble methods."""
    
    def __init__(self, config: MultiApproachConfig = None):
        """Initialize multi-approach material grouper."""
        self.config = config or MultiApproachConfig()
        self.preprocessor = EnhancedTextPreprocessor()
        self.similarity_calc = MultiApproachSimilarityCalculator(self.config)
        self.ensemble_clusterer = EnsembleClusterer(self.config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preprocessing."""
        self.logger.info("Multi-approach preprocessing...")
        
        processed_df = df.copy()
        
        # Add unique ID if not present
        if 'ID' not in processed_df.columns:
            processed_df['ID'] = range(1, len(processed_df) + 1)
        
        # Fill missing values
        processed_df['MTL_DESC'] = processed_df['MTL_DESC'].fillna('')
        processed_df['ITEM_DESCRIPTION'] = processed_df['ITEM_DESCRIPTION'].fillna('')
        processed_df['PSL_UNIT'] = processed_df['PSL_UNIT'].fillna('NO')
        
        # Combine descriptions
        processed_df['combined_desc'] = (
            processed_df['MTL_DESC'].astype(str) + ' ' + 
            processed_df['ITEM_DESCRIPTION'].astype(str)
        )
        
        # Enhanced text processing
        processed_df['processed_text'] = processed_df['combined_desc'].apply(
            self.preprocessor.extract_enhanced_keywords
        )
        
        # Extract material types
        processed_df['material_type'] = processed_df['combined_desc'].apply(
            self.preprocessor.extract_material_type
        )
        
        # Remove empty processed texts
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0].copy()
        
        self.logger.info(f"Multi-approach preprocessing completed: {len(processed_df)} records")
        return processed_df
    
    def build_multi_approach_similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build similarity matrix using multiple approaches."""
        self.logger.info("Building multi-approach similarity matrix...")
        
        # Prepare numerical features
        numerical_features = self._prepare_numerical_features(df)
        
        # Fit all models
        self.similarity_calc.fit_all_models(df['processed_text'].tolist(), numerical_features)
        
        # Build similarity matrix
        n_items = len(df)
        similarity_matrix = np.zeros((n_items, n_items))
        
        for i in tqdm(range(n_items), desc="Computing multi-approach similarities"):
            for j in range(i + 1, n_items):
                similarity = self.similarity_calc.calculate_multi_approach_similarity(
                    i, j,
                    df.iloc[i]['processed_text'],
                    df.iloc[j]['processed_text'],
                    df.iloc[i]['PSL_UNIT'],
                    df.iloc[j]['PSL_UNIT']
                )
                
                # Boost similarity for same material types
                if df.iloc[i]['material_type'] == df.iloc[j]['material_type']:
                    similarity *= 1.1
                
                similarity_matrix[i, j] = min(similarity, 1.0)
                similarity_matrix[j, i] = similarity_matrix[i, j]
        
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _prepare_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare numerical features for GNN."""
        features = []
        
        # Rate features
        features.append(df['RATE'].values)
        features.append(np.log1p(df['RATE'].values))  # Log-transformed rate
        
        # Text length features
        features.append(df['MTL_DESC'].str.len().fillna(0).values)
        features.append(df['ITEM_DESCRIPTION'].str.len().fillna(0).values)
        
        # Categorical features (encoded)
        le_store = LabelEncoder()
        features.append(le_store.fit_transform(df['STORE_CODE'].astype(str)))
        
        le_unit = LabelEncoder()
        features.append(le_unit.fit_transform(df['PSL_UNIT'].astype(str)))
        
        return np.column_stack(features)
    
    def detect_outliers(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Detect outliers using multiple methods."""
        if not self.config.use_outlier_detection:
            return np.array([])
        
        # Use similarity matrix features for outlier detection
        features = []
        for i in range(similarity_matrix.shape[0]):
            row_similarities = similarity_matrix[i, :]
            features.append([
                np.mean(row_similarities),
                np.max(row_similarities[row_similarities < 1.0]),
                np.std(row_similarities),
                np.percentile(row_similarities, 75),
                np.percentile(row_similarities, 25)
            ])
        
        features = np.array(features)
        outlier_labels = self.outlier_detector.fit_predict(features)
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        self.logger.info(f"Detected {len(outlier_indices)} outliers")
        return outlier_indices
    
    def process_materials(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main processing method using multi-approach techniques."""
        start_time = datetime.now()
        self.logger.info("Starting multi-approach material grouping...")
        
        # Preprocessing
        processed_df = self.preprocess_data(df)
        
        # Build similarity matrix
        similarity_matrix = self.build_multi_approach_similarity_matrix(processed_df)
        
        # Detect outliers
        outlier_indices = self.detect_outliers(similarity_matrix)
        
        # Perform ensemble clustering
        cluster_labels = self.ensemble_clusterer.fit_predict_ensemble(similarity_matrix)
        
        # Add results to dataframe
        processed_df['cluster_id'] = cluster_labels
        processed_df['is_outlier'] = False
        processed_df.loc[outlier_indices, 'is_outlier'] = True
        
        # Generate representatives
        representatives = self._generate_representatives(processed_df, cluster_labels, similarity_matrix)
        
        # Mark representatives
        processed_df['is_representative'] = False
        for cluster_id, rep_info in representatives.items():
            rep_idx = rep_info['representative_index']
            mask = processed_df.index == rep_idx
            processed_df.loc[mask, 'is_representative'] = True
        
        # Generate report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        report = self._generate_comprehensive_report(
            df, processed_df, representatives, processing_time, 
            outlier_indices, similarity_matrix
        )
        
        self.logger.info(f"Multi-approach processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Found {len(representatives)} clusters")
        
        return processed_df, report
    
    def _generate_representatives(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                                similarity_matrix: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Generate cluster representatives."""
        representatives = {}
        
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 1:
                rep_idx = cluster_indices[0]
                cluster_quality = 1.0
            else:
                rep_idx, cluster_quality = self._find_best_representative(
                    cluster_indices, similarity_matrix, df
                )
            
            rate_values = cluster_data['RATE'].values
            cluster_id_str = str(cluster_id)
            
            representatives[cluster_id_str] = {
                'representative_index': int(rep_idx),
                'mtl_desc': str(df.iloc[rep_idx]['MTL_DESC']),
                'item_description': str(df.iloc[rep_idx]['ITEM_DESCRIPTION']),
                'psl_unit': str(df.iloc[rep_idx]['PSL_UNIT']),
                'material_type': str(df.iloc[rep_idx]['material_type']),
                'rate': float(df.iloc[rep_idx]['RATE']),
                'store_code': int(df.iloc[rep_idx]['STORE_CODE']),
                'group_size': int(len(cluster_data)),
                'cluster_quality': float(cluster_quality),
                'total_rate': float(rate_values.sum()),
                'avg_rate': float(rate_values.mean()),
                'median_rate': float(np.median(rate_values)),
                'rate_std': float(rate_values.std()),
                'rate_range': float(rate_values.max() - rate_values.min()),
                'store_codes': [int(x) for x in cluster_data['STORE_CODE'].unique()],
                'material_types': [str(x) for x in cluster_data['material_type'].unique()]
            }
        
        return representatives
    
    def _find_best_representative(self, cluster_indices: np.ndarray, 
                                similarity_matrix: np.ndarray,
                                df: pd.DataFrame) -> Tuple[int, float]:
        """Find the best representative using multiple criteria."""
        if len(cluster_indices) == 1:
            return cluster_indices[0], 1.0
        
        scores = []
        
        for i, idx in enumerate(cluster_indices):
            similarities = [similarity_matrix[idx, other_idx] 
                          for j, other_idx in enumerate(cluster_indices) if i != j]
            avg_similarity = np.mean(similarities)
            
            cluster_similarities = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            centroid_similarity = np.mean(cluster_similarities[i, :])
            
            cluster_rates = df.iloc[cluster_indices]['RATE'].values
            rate_score = 1 - abs(df.iloc[idx]['RATE'] - np.median(cluster_rates)) / (
                np.max(cluster_rates) - np.min(cluster_rates) + 1e-6)
            
            combined_score = 0.5 * avg_similarity + 0.3 * centroid_similarity + 0.2 * rate_score
            scores.append(combined_score)
        
        best_idx = np.argmax(scores)
        cluster_quality = np.mean([similarity_matrix[cluster_indices[i], cluster_indices[j]] 
                                 for i in range(len(cluster_indices)) 
                                 for j in range(i + 1, len(cluster_indices))])
        
        return cluster_indices[best_idx], cluster_quality
    
    def _generate_comprehensive_report(self, original_df: pd.DataFrame, 
                                     processed_df: pd.DataFrame,
                                     representatives: Dict[str, Dict[str, Any]],
                                     processing_time: float,
                                     outlier_indices: np.ndarray,
                                     similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive report."""
        
        avg_cluster_quality = np.mean([rep['cluster_quality'] for rep in representatives.values()])
        material_type_dist = processed_df['material_type'].value_counts().to_dict()
        
        report = {
            'processing_time_seconds': float(processing_time),
            'original_records': int(len(original_df)),
            'processed_records': int(len(processed_df)),
            'total_clusters': int(len(representatives)),
            'reduction_percentage': float((1 - len(representatives) / len(processed_df)) * 100),
            'avg_cluster_quality': float(avg_cluster_quality),
            'outliers_detected': int(len(outlier_indices)),
            'material_type_distribution': {str(k): int(v) for k, v in material_type_dist.items()},
            'approaches_used': {
                'slm': self.config.use_slm,
                'gnn': self.config.use_gnn,
                'ensemble': self.config.use_ensemble,
                'ensemble_methods': self.config.ensemble_methods
            },
            'model_configuration': {
                'slm_model': self.config.slm_model_name if self.config.use_slm else None,
                'gnn_layers': self.config.gnn_num_layers if self.config.use_gnn else None,
                'ensemble_methods': self.config.ensemble_methods,
                'weights': {
                    'slm_weight': self.config.slm_weight,
                    'gnn_weight': self.config.gnn_weight,
                    'tfidf_weight': self.config.tfidf_weight,
                    'fuzzy_weight': self.config.fuzzy_weight
                }
            },
            'group_statistics': {
                'min_group_size': int(min(rep['group_size'] for rep in representatives.values())),
                'max_group_size': int(max(rep['group_size'] for rep in representatives.values())),
                'avg_group_size': float(np.mean([rep['group_size'] for rep in representatives.values()])),
                'single_item_groups': int(sum(1 for rep in representatives.values() if rep['group_size'] == 1))
            },
            'representatives': representatives
        }
        
        return report

def create_comprehensive_visualizations(df: pd.DataFrame, report: Dict[str, Any], 
                                       output_dir: str = "multi_approach_output"):
    """Create comprehensive visualizations for multi-approach results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Approach Comparison Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Approach Material Grouping Dashboard', fontsize=20, fontweight='bold')
    
    # Approach usage
    ax1 = axes[0, 0]
    approaches = ['SLM', 'GNN', 'TF-IDF', 'Fuzzy', 'Ensemble']
    weights = [report['model_configuration']['weights']['slm_weight'],
              report['model_configuration']['weights']['gnn_weight'],
              report['model_configuration']['weights']['tfidf_weight'],
              report['model_configuration']['weights']['fuzzy_weight'],
              0.1]  # Ensemble weight
    
    bars = ax1.bar(approaches, weights, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#feca57', '#96ceb4'])
    ax1.set_title('Approach Weights in Ensemble', fontweight='bold')
    ax1.set_ylabel('Weight')
    
    for bar, weight in zip(bars, weights):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Cluster distribution
    ax2 = axes[0, 1]
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    ax2.bar(range(len(cluster_counts)), cluster_counts.values, color='lightblue')
    ax2.set_title('Cluster Size Distribution', fontweight='bold')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Materials')
    
    # Processing efficiency
    ax3 = axes[0, 2]
    efficiency_metrics = ['Processing\nTime (sec)', 'Reduction\nPercentage', 'Cluster\nQuality']
    efficiency_values = [report['processing_time_seconds'], 
                        report['reduction_percentage'], 
                        report['avg_cluster_quality'] * 100]
    
    ax3.bar(efficiency_metrics, efficiency_values, color=['orange', 'green', 'purple'])
    ax3.set_title('Processing Efficiency Metrics', fontweight='bold')
    ax3.set_ylabel('Value')
    
    # Material type distribution
    ax4 = axes[1, 0]
    material_dist = report['material_type_distribution']
    if material_dist:
        materials = list(material_dist.keys())
        counts = list(material_dist.values())
        
        wedges, texts, autotexts = ax4.pie(counts, labels=materials, autopct='%1.1f%%',
                                          startangle=90, colors=sns.color_palette("Set3"))
        ax4.set_title('Material Type Distribution', fontweight='bold')
    
    # Outlier analysis
    ax5 = axes[1, 1]
    outlier_count = report['outliers_detected']
    normal_count = report['processed_records'] - outlier_count
    
    labels = ['Normal Materials', 'Outliers']
    sizes = [normal_count, outlier_count]
    colors = ['lightblue', 'red']
    
    ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Outlier Detection Results', fontweight='bold')
    
    # Performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
Multi-Approach Results Summary:

• Total Records: {report['original_records']:,}
• Clusters Created: {report['total_clusters']:,}
• Reduction: {report['reduction_percentage']:.1f}%
• Processing Time: {report['processing_time_seconds']:.2f}s
• Cluster Quality: {report['avg_cluster_quality']:.3f}
• Outliers: {report['outliers_detected']}

Approaches Used:
• SLM: {report['approaches_used']['slm']}
• GNN: {report['approaches_used']['gnn']}
• Ensemble: {report['approaches_used']['ensemble']}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / f'multi_approach_dashboard_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interactive 3D Visualization
    if 'embeddings' in df.columns:
        try:
            embeddings = np.array([eval(emb) if isinstance(emb, str) else emb 
                                 for emb in df['embeddings']])
            
            # Use PCA for 3D visualization
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(embeddings)
            
            fig = go.Figure(data=go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=df['cluster_id'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster ID")
                ),
                text=df['ID'] if 'ID' in df.columns else df.index,
                textposition="top center",
                hovertemplate='<b>ID:</b> %{text}<br>' +
                             '<b>Cluster:</b> %{marker.color}<br>' +
                             '<b>X:</b> %{x:.2f}<br>' +
                             '<b>Y:</b> %{y:.2f}<br>' +
                             '<b>Z:</b> %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='3D Multi-Approach Clustering Visualization',
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                width=1000,
                height=700
            )
            
            fig.write_html(output_path / f'3d_clustering_visualization_{timestamp}.html')
        except:
            pass
    
    print(f"\nComprehensive visualizations saved to {output_path}")

def save_multi_approach_results(output_df: pd.DataFrame, report: Dict[str, Any], 
                               output_dir: str = "multi_approach_output") -> str:
    """Save multi-approach results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results
    csv_filename = f"multi_approach_grouped_materials_{timestamp}.csv"
    csv_path = output_path / csv_filename
    output_df.to_csv(csv_path, index=False)
    
    # Save representatives only
    representatives_df = output_df[output_df['is_representative']].copy()
    rep_filename = f"multi_approach_representatives_{timestamp}.csv"
    rep_path = output_path / rep_filename
    representatives_df.to_csv(rep_path, index=False)
    
    # Save detailed report
    report_filename = f"multi_approach_report_{timestamp}.json"
    report_path = output_path / report_filename
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nMulti-approach results saved:")
    print(f"  - Full results: {csv_path}")
    print(f"  - Representatives: {rep_path}")
    print(f"  - Report: {report_path}")
    
    return str(csv_path)

def main():
    """Main function for multi-approach material grouping."""
    
    input_file = "sample.csv"    # replace the original csv dataset here
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
        
        # Multi-approach configuration
        config = MultiApproachConfig(
            slm_model_name="all-MiniLM-L6-v2",
            use_slm=True,
            use_gnn=True,
            use_ensemble=True,
            ensemble_methods=['dbscan', 'hierarchical', 'spectral'],
            slm_weight=0.3,
            gnn_weight=0.3,
            tfidf_weight=0.2,
            fuzzy_weight=0.2,
            eps=0.25,
            min_samples=2,
            use_adaptive_threshold=True,
            use_outlier_detection=True
        )
        
        # Initialize and run multi-approach grouper
        grouper = MultiApproachMaterialGrouper(config)
        output_df, report = grouper.process_materials(df)
        
        # Save results
        final_csv_path = save_multi_approach_results(output_df, report)
        
        # Create visualizations
        create_comprehensive_visualizations(output_df, report)
        
        # Print comprehensive summary
        print(f"\n{'='*70}")
        print("MULTI-APPROACH MATERIAL GROUPING SUMMARY")
        print(f"{'='*70}")
        print(f"Original records: {report['original_records']:,}")
        print(f"Processed records: {report['processed_records']:,}")
        print(f"Total clusters: {report['total_clusters']:,}")
        print(f"Reduction achieved: {report['reduction_percentage']:.1f}%")
        print(f"Processing time: {report['processing_time_seconds']:.2f} seconds")
        print(f"Average cluster quality: {report['avg_cluster_quality']:.3f}")
        print(f"Outliers detected: {report['outliers_detected']}")
        
        print(f"\nApproaches Used:")
        print(f"  - SLM (Sentence Transformers): {report['approaches_used']['slm']}")
        print(f"  - GNN (Graph Neural Networks): {report['approaches_used']['gnn']}")
        print(f"  - Ensemble Clustering: {report['approaches_used']['ensemble']}")
        print(f"  - Ensemble Methods: {', '.join(report['approaches_used']['ensemble_methods'])}")
        
        print(f"\nModel Weights:")
        weights = report['model_configuration']['weights']
        for approach, weight in weights.items():
            print(f"  - {approach}: {weight}")
        
        print(f"\nCluster Statistics:")
        stats = report['group_statistics']
        print(f"  - Average cluster size: {stats['avg_group_size']:.1f}")
        print(f"  - Largest cluster: {stats['max_group_size']} items")
        print(f"  - Single-item clusters: {stats['single_item_groups']}")
        
        return output_df, report
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result_df, report = main()
