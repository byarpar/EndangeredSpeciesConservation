import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import (
    cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, RFE
)
from sklearn.inspection import permutation_importance
import os
import joblib
from time import time
# Try to import shap, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP visualizations will be disabled.")
import matplotlib.colors as mcolors

# Dictionary of available models
AVAILABLE_MODELS = {
    # Classification models
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "svm_classifier": SVC,
    "knn_classifier": KNeighborsClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    "mlp_classifier": MLPClassifier,
    "logistic_regression": LogisticRegression,
    
    # Regression models
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    "linear_regression": LinearRegression,
    "ridge_regression": Ridge,
    "lasso_regression": Lasso,
    "elastic_net": ElasticNet,
    "svm_regressor": SVR,
    "knn_regressor": KNeighborsRegressor,
    "decision_tree_regressor": DecisionTreeRegressor,
    "mlp_regressor": MLPRegressor
}

# Default hyperparameters for each model
DEFAULT_PARAMS = {
    # Classification models
    "random_forest_classifier": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "gradient_boosting_classifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    },
    "svm_classifier": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True
    },
    "knn_classifier": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "decision_tree_classifier": {
        "max_depth": 10,
        "min_samples_split": 5
    },
    "mlp_classifier": {
        "hidden_layer_sizes": (100,),
        "max_iter": 300,
        "alpha": 0.0001
    },
    "logistic_regression": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 200
    },
    
    # Regression models
    "random_forest_regressor": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "gradient_boosting_regressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    },
    "linear_regression": {},
    "ridge_regression": {
        "alpha": 1.0
    },
    "lasso_regression": {
        "alpha": 0.1
    },
    "elastic_net": {
        "alpha": 0.1,
        "l1_ratio": 0.5
    },
    "svm_regressor": {
        "C": 1.0,
        "kernel": "rbf"
    },
    "knn_regressor": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "decision_tree_regressor": {
        "max_depth": 10,
        "min_samples_split": 5
    },
    "mlp_regressor": {
        "hidden_layer_sizes": (100,),
        "max_iter": 300,
        "alpha": 0.0001
    }
}

def build_model(model_type="random_forest_classifier", random_state=42, **kwargs):
    """
    Build a machine learning model for species conservation.
    
    In a professional conservation setting, these models would be used to:
    - Predict conservation status for newly assessed species
    - Identify key factors driving species vulnerability
    - Prioritize conservation efforts based on vulnerability predictions
    - Evaluate the effectiveness of conservation interventions
    
    Args:
        model_type (str): Type of model to build. See AVAILABLE_MODELS for options.
        random_state (int): Random seed for reproducibility
        **kwargs: Additional parameters to pass to the model constructor
        
    Returns:
        model: The initialized model
    """
    print(f"Building {model_type} model...")

    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    # Get the model class
    model_class = AVAILABLE_MODELS[model_type]
    
    # Get default parameters for this model type
    params = DEFAULT_PARAMS.get(model_type, {}).copy()
    
    # Update with any provided parameters
    params.update(kwargs)
    
    # Add random_state if the model supports it
    if "random_state" in model_class.__init__.__code__.co_varnames:
        params["random_state"] = random_state
    
    # Create and return the model
    return model_class(**params)

def train_model(model, X_train, y_train, X_test=None, y_test=None):
    """
    Train the machine learning model.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features (optional)
        y_test: Test targets (optional)
        
    Returns:
        model: The trained model
    """
    print("Training the machine learning model...")
    start_time = time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        
        # Check if this is a classifier or regressor
        if hasattr(model, "classes_") or hasattr(model, "predict_proba"):
            # Classification model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test accuracy: {accuracy:.4f}")
        else:
            # Regression model
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Test R² score: {r2:.4f}")
            print(f"Test MSE: {mse:.4f}")

    return model

def evaluate_model(model, X_test, y_test, status_map=None, feature_names=None):
    """
    Evaluate model performance with appropriate metrics.
    
    In a conservation context, model evaluation helps:
    - Assess the reliability of predictions for decision-making
    - Understand model strengths and limitations
    - Identify areas where additional data collection is needed
    - Communicate model performance to stakeholders and policymakers
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test targets
        status_map: Mapping of status values to labels (for classification)
        feature_names: List of feature names for importance plots
        
    Returns:
        dict: Dictionary of evaluation results
    """
    print("Evaluating model performance...")

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check if this is a classifier or regressor
    is_classifier = hasattr(model, "classes_") or hasattr(model, "predict_proba")
    
    if is_classifier:
        # Classification model
        cm = confusion_matrix(y_test, y_pred)
        
        # Reverse status map for display
        if status_map:
            status_labels = {v: k for k, v in status_map.items()}
            
            # Get unique classes in the test set and predictions
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            unique_classes = sorted(unique_classes)  # Sort to ensure consistent order
            
            # Create labels only for classes that appear in the data
            labels = [status_labels[i] for i in unique_classes]
            
            # Print classification report with explicit labels parameter
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, labels=unique_classes, target_names=labels))
            
            # Print confusion matrix
            print("\nConfusion Matrix:")
            print(cm)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix - Conservation Status Prediction')
            plt.ylabel('True Status')
            plt.xlabel('Predicted Status')
            
            # Save the plot
            plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{type(model).__name__}.png'))
            plt.close()
            
            # ROC curve and AUC if the model supports predict_proba
            if hasattr(model, "predict_proba"):
                try:
                    # For multi-class, we'll use one-vs-rest approach
                    y_prob = model.predict_proba(X_test)
                    
                    plt.figure(figsize=(10, 8))
                    
                    # If binary classification
                    if len(unique_classes) == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC)')
                        plt.legend(loc="lower right")
                    else:
                        # For multi-class, plot ROC for each class
                        for i, class_idx in enumerate(unique_classes):
                            y_test_binary = (y_test == class_idx).astype(int)
                            fpr, tpr, _ = roc_curve(y_test_binary, y_prob[:, i])
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, lw=2, 
                                    label=f'{labels[i]} (AUC = {roc_auc:.2f})')
                        
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('ROC Curves for Conservation Status Prediction')
                        plt.legend(loc="lower right")
                    
                    plt.savefig(os.path.join(plots_dir, f'roc_curve_{type(model).__name__}.png'))
                    plt.close()
                except Exception as e:
                    print(f"Could not generate ROC curve: {e}")
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            
            if feature_names and len(feature_importances) == len(feature_names):
                # Plot feature importances
                plot_feature_importance(model, feature_names)
        
        # For linear models, get coefficients
        coefficients = None
        if hasattr(model, "coef_"):
            coefficients = model.coef_
            
            if feature_names and coefficients.ndim == 1 and len(coefficients) == len(feature_names):
                # Plot coefficients
                plot_feature_importance(model, feature_names, plot_type='coefficients')
        
        # If it's a decision tree, visualize it
        if isinstance(model, DecisionTreeClassifier) and feature_names:
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_names, class_names=labels if status_map else None, 
                     filled=True, rounded=True, fontsize=10)
            plt.title('Decision Tree for Conservation Status Prediction')
            plt.savefig(os.path.join(plots_dir, 'decision_tree_visualization.png'), bbox_inches='tight')
            plt.close()
        
        # Generate SHAP values for model interpretability
        if SHAP_AVAILABLE:
            try:
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                    # Create a small sample for SHAP analysis (for efficiency)
                    sample_size = min(100, X_test.shape[0])
                    X_sample = X_test[:sample_size]
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Plot SHAP summary
                    plt.figure(figsize=(12, 8))
                    if isinstance(shap_values, list):  # For multi-class
                        # Combine shap values for all classes
                        shap_values_combined = np.abs(np.array(shap_values)).mean(0)
                        shap.summary_plot(shap_values_combined, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                    
                    plt.title('SHAP Feature Importance for Conservation Status Prediction')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'shap_importance_{type(model).__name__}.png'))
                    plt.close()
            except Exception as e:
                print(f"Could not generate SHAP values: {e}")
        else:
            print("Skipping SHAP analysis: shap package not installed")
        
        return {
            'y_pred': y_pred,
            'confusion_matrix': cm,
            'labels': labels if status_map else None,
            'feature_importances': feature_importances,
            'coefficients': coefficients,
            'model_type': 'classifier',
            'accuracy': accuracy_score(y_test, y_pred)
        }
    else:
        # Regression model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nRegression Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        
        # Create a colormap based on conservation status if available
        if 'conservation_status' in locals():
            status_to_num = {
                'Critically Endangered': 4,
                'Endangered': 3,
                'Vulnerable': 2,
                'Near Threatened': 1,
                'Least Concern': 0
            }
            colors = [status_to_num.get(status, 0) for status in conservation_status]
            plt.scatter(y_test, y_pred, c=colors, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(label='Conservation Status')
        else:
            plt.scatter(y_test, y_pred, alpha=0.7)
            
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Vulnerability Score')
        plt.ylabel('Predicted Vulnerability Score')
        plt.title('Actual vs Predicted Vulnerability Scores')
        
        # Add R² value to the plot
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Save the plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, f'actual_vs_predicted_{type(model).__name__}.png'))
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='--')
        plt.xlabel('Predicted Vulnerability Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot for Vulnerability Prediction')
        plt.savefig(os.path.join(plots_dir, f'residuals_{type(model).__name__}.png'))
        plt.close()
        
        # Get feature importances if available (for Random Forest Regressor)
        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            
            if feature_names and len(feature_importances) == len(feature_names):
                # Plot feature importances
                plot_feature_importance(model, feature_names)
        
        # For linear models, get coefficients
        coefficients = None
        if hasattr(model, "coef_"):
            coefficients = model.coef_
            
            if feature_names and (coefficients.ndim == 1 or coefficients.shape[0] == 1) and len(coefficients.flatten()) == len(feature_names):
                # Plot coefficients
                plot_feature_importance(model, feature_names, plot_type='coefficients')
        
        # If it's a decision tree, visualize it
        if isinstance(model, DecisionTreeRegressor) and feature_names:
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
            plt.title('Decision Tree for Vulnerability Score Prediction')
            plt.savefig(os.path.join(plots_dir, 'decision_tree_regression_visualization.png'), bbox_inches='tight')
            plt.close()
        
        # Generate SHAP values for model interpretability
        if SHAP_AVAILABLE:
            try:
                if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                    # Create a small sample for SHAP analysis (for efficiency)
                    sample_size = min(100, X_test.shape[0])
                    X_sample = X_test[:sample_size]
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Plot SHAP summary
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                    plt.title('SHAP Feature Importance for Vulnerability Score Prediction')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'shap_importance_{type(model).__name__}.png'))
                    plt.close()
            except Exception as e:
                print(f"Could not generate SHAP values: {e}")
        else:
            print("Skipping SHAP analysis: shap package not installed")
        
        return {
            'y_pred': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importances': feature_importances,
            'coefficients': coefficients,
            'model_type': 'regressor'
        }

def plot_feature_importance(model, feature_names, plot_type='importance'):
    """
    Plot feature importance or coefficients from the model.
    
    In conservation science, understanding feature importance helps:
    - Identify key drivers of species vulnerability
    - Focus conservation efforts on the most impactful factors
    - Design targeted interventions that address specific threats
    - Inform policy decisions and resource allocation
    
    Args:
        model: The trained model
        feature_names: List of feature names
        plot_type: Type of plot ('importance' for tree-based models, 'coefficients' for linear models)
        
    Returns:
        indices: Sorted indices of features by importance
    """
    plt.figure(figsize=(12, 8))
    
    if plot_type == 'importance' and hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'Key Factors Influencing Species Conservation - {type(model).__name__}')
        plt.bar(range(len(importances)), importances[indices], align='center', color='darkgreen')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Environmental and Biological Factors')
        plt.ylabel('Relative Importance')
        plt.tight_layout()
        
    elif plot_type == 'coefficients' and hasattr(model, 'coef_'):
        # For linear models like Linear Regression
        coefficients = model.coef_
        
        # For multi-output regression or multi-class classification, take the mean of coefficients
        if coefficients.ndim > 1 and coefficients.shape[0] > 1:
            coefficients = np.mean(coefficients, axis=0)
        elif coefficients.ndim > 1 and coefficients.shape[0] == 1:
            coefficients = coefficients.flatten()
            
        # Get absolute values and sort
        abs_coefficients = np.abs(coefficients)
        indices = np.argsort(abs_coefficients)[::-1]
        
        # Create a color map based on coefficient sign
        colors = ['darkred' if c < 0 else 'darkgreen' for c in coefficients[indices]]
        
        plt.title(f'Key Factors Influencing Species Conservation - {type(model).__name__}')
        plt.bar(range(len(coefficients)), abs_coefficients[indices], align='center', color=colors)
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Environmental and Biological Factors')
        plt.ylabel('Absolute Coefficient Magnitude')
        
        # Add a legend for positive/negative impact
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='Positive Impact (Decreases Vulnerability)'),
            Patch(facecolor='darkred', label='Negative Impact (Increases Vulnerability)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
    else:
        print("No feature importance or coefficients available for this model")
        return None
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    model_type = type(model).__name__
    plt.savefig(os.path.join(plots_dir, f'feature_importance_{model_type}.png'))
    plt.close()
    
    return indices

def compare_models(models, X_test, y_test, is_classification=True):
    """
    Compare multiple models on the same test data.
    
    In conservation planning, model comparison helps:
    - Select the most reliable model for decision support
    - Understand the trade-offs between different modeling approaches
    - Assess prediction uncertainty across multiple models
    - Build consensus from multiple modeling perspectives
    
    Args:
        models: Dictionary of trained models {name: model}
        X_test: Test features
        y_test: Test targets
        is_classification: Whether this is a classification task
        
    Returns:
        dict: Dictionary of comparison results
    """
    print("\nComparing models for conservation decision support:")
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        y_pred = model.predict(X_test)
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'model': model
            }
            print(f"Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {
                'mse': mse,
                'r2': r2,
                'model': model
            }
            print(f"MSE: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    if is_classification:
        # Plot accuracy for classification models
        accuracies = [results[name]['accuracy'] for name in results]
        plt.bar(results.keys(), accuracies, color='darkgreen')
        plt.title('Model Comparison - Conservation Status Prediction Accuracy')
        plt.ylabel('Accuracy')
        
        # Add values on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
            
    else:
        # Plot R² for regression models
        r2_scores = [results[name]['r2'] for name in results]
        plt.bar(results.keys(), r2_scores, color='darkblue')
        plt.title('Model Comparison - Vulnerability Score Prediction Accuracy')
        plt.ylabel('R² Score')
        
        # Add values on top of bars
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.xlabel('Model')
    plt.ylim(0, 1.1 if is_classification else max(r2_scores) * 1.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    task_type = 'classification' if is_classification else 'regression'
    plt.savefig(os.path.join(plots_dir, f'model_comparison_{task_type}.png'))
    plt.close()
    
    return results

def perform_cross_validation(model, X, y, cv=5, scoring=None):
    """
    Perform cross-validation on a model.
    
    In conservation applications, cross-validation helps:
    - Ensure models are robust across different subsets of species
    - Reduce overfitting to particular regions or taxonomic groups
    - Provide confidence intervals for model performance metrics
    - Test model generalizability to new species or regions
    
    Args:
        model: The model to evaluate
        X: Features
        y: Targets
        cv: Number of cross-validation folds
        scoring: Scoring metric (None uses default for the model type)
        
    Returns:
        dict: Dictionary of cross-validation results
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    # Determine if this is a classifier or regressor
    is_classifier = hasattr(model, "classes_") or hasattr(model, "predict_proba")
    
    # Set default scoring if not provided
    if scoring is None:
        scoring = 'accuracy' if is_classifier else 'r2'
    
    # Perform cross-validation
    start_time = time()
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    cv_time = time() - start_time
    
    print(f"Cross-validation completed in {cv_time:.2f} seconds")
    print(f"CV {scoring} scores: {cv_scores}")
    print(f"Mean CV {scoring} score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, cv+1), cv_scores, color='darkblue')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='-', label=f'Mean: {cv_scores.mean():.4f}')
    plt.axhline(y=cv_scores.mean() + cv_scores.std(), color='r', linestyle='--', 
               label=f'Std Dev: ±{cv_scores.std():.4f}')
    plt.axhline(y=cv_scores.mean() - cv_scores.std(), color='r', linestyle='--')
    plt.xlabel('Fold')
    plt.ylabel(scoring.capitalize())
    plt.title(f'Cross-Validation Results ({cv}-fold)')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, f'cross_validation_{type(model).__name__}.png'))
    plt.close()
    
    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'scoring': scoring
    }

def tune_hyperparameters(model, X, y, param_grid, cv=5, scoring=None, n_jobs=-1, method='grid'):
    """
    Tune hyperparameters using grid search or randomized search.
    
    In conservation applications, hyperparameter tuning helps:
    - Optimize model performance for specific conservation metrics
    - Balance model complexity with interpretability
    - Adapt models to the specific characteristics of ecological data
    - Improve prediction accuracy for rare or endangered species
    
    Args:
        model: The model to tune
        X: Features
        y: Targets
        param_grid: Dictionary of parameters to search
        cv: Number of cross-validation folds
        scoring: Scoring metric (None uses default for the model type)
        n_jobs: Number of parallel jobs (-1 for all processors)
        method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        
    Returns:
        dict: Dictionary of tuning results
    """
    # Determine if this is a classifier or regressor
    is_classifier = hasattr(model, "classes_") or hasattr(model, "predict_proba")
    
    # Set default scoring if not provided
    if scoring is None:
        scoring = 'accuracy' if is_classifier else 'r2'
    
    print(f"\nPerforming hyperparameter tuning using {method} search...")
    print(f"Scoring metric: {scoring}")
    
    # Create the search object
    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1, n_iter=20)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'.")
    
    # Perform the search
    start_time = time()
    search.fit(X, y)
    tuning_time = time() - start_time
    
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best {scoring} score: {search.best_score_:.4f}")
    
    # Create a model with the best parameters
    best_model = model.__class__(**search.best_params_)
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_model': best_model,
        'cv_results': search.cv_results_,
        'scoring': scoring
    }

def select_features(X, y, k=5, method='k_best', is_classification=True):
    """
    Select the most important features.
    
    In conservation, feature selection helps:
    - Identify the most influential factors driving species decline
    - Reduce data dimensionality and model complexity
    - Improve model interpretability and generalization
    - Focus data collection efforts on the most relevant variables
    
    Args:
        X: Features
        y: Targets
        k: Number of features to select
        method: Method to use ('k_best' or 'rfe')
        is_classification: Whether this is a classification task
        
    Returns:
        dict: Dictionary of feature selection results
    """
    print(f"\nPerforming feature selection using {method}...")
    
    if method == 'k_best':
        # Use SelectKBest
        if is_classification:
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
        
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        return {
            'selector': selector,
            'X_new': X_new,
            'selected_indices': selected_indices,
            'scores': selector.scores_
        }
    
    elif method == 'rfe':
        # Use Recursive Feature Elimination
        if is_classification:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=k, step=1)
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        return {
            'selector': selector,
            'X_new': X_new,
            'selected_indices': selected_indices,
            'ranking': selector.ranking_
        }
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'k_best' or 'rfe'.")

def create_ensemble(models, ensemble_type='voting', is_classification=True, **kwargs):
    """
    Create an ensemble of models.
    
    In conservation, ensemble modeling helps:
    - Improve prediction accuracy and robustness
    - Reduce model uncertainty and bias
    - Combine the strengths of different modeling approaches
    - Provide more reliable decision support for conservation planning
    
    Args:
        models: List of (name, model) tuples
        ensemble_type: Type of ensemble ('voting' or 'stacking')
        is_classification: Whether this is a classification task
        **kwargs: Additional parameters for the ensemble
        
    Returns:
        model: The ensemble model
    """
    print(f"\nCreating {ensemble_type} ensemble...")
    
    if ensemble_type == 'voting':
        # Create a voting ensemble
        if is_classification:
            ensemble = VotingClassifier(estimators=models, **kwargs)
        else:
            ensemble = VotingRegressor(estimators=models, **kwargs)
    
    elif ensemble_type == 'stacking':
        # Create a stacking ensemble
        if is_classification:
            ensemble = StackingClassifier(estimators=models, **kwargs)
        else:
            ensemble = StackingRegressor(estimators=models, **kwargs)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}. Use 'voting' or 'stacking'.")
    
    return ensemble

def save_model(model, model_name='species_conservation_model.pkl'):
    """Save the trained model."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_path = os.path.join(models_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """Load a trained model."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_scaler(scaler, scaler_name='species_scaler.pkl'):
    """Save the feature scaler."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    scaler_path = os.path.join(models_dir, scaler_name)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    return scaler_path
