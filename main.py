# main.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data_processing import generate_synthetic_data, preprocess_data, save_data, load_data
from src.model import (
    build_model, train_model, evaluate_model, plot_feature_importance, 
    save_model, save_scaler, compare_models, perform_cross_validation,
    tune_hyperparameters, select_features, create_ensemble
)
from src.conservation_analysis import (
    calculate_conservation_priority_score, identify_priority_species,
    estimate_conservation_resources, cluster_species_by_traits,
    generate_conservation_report, plot_conservation_priority_map,
    create_species_profile
)

def main():
    try:
        print("=== Enviro Wise Endangered Species Conservation Project ===\n")
        
        # Step 1: Data Collection and Preprocessing
        print("Step 1: Data Collection and Preprocessing")
        data = load_data('species_data.csv')
        
        # Preprocess for classification
        preprocessed_data_clf = preprocess_data(data, task_type="classification")
        
        # Preprocess for regression
        preprocessed_data_reg = preprocess_data(data, task_type="regression")
        
        # Step 2: Conservation Analysis
        print("\nStep 2: Conservation Analysis")
        
        # Calculate conservation priority scores
        data_with_priority = calculate_conservation_priority_score(data)
        
        # Identify priority species
        priority_species = identify_priority_species(data_with_priority)
        print(f"Identified {len(priority_species)} priority species for conservation")
        
        # Estimate conservation resources
        data_with_resources = estimate_conservation_resources(data_with_priority)
        
        # Cluster species by traits
        clustered_data, kmeans = cluster_species_by_traits(data_with_resources)
        
        # Save enhanced data
        save_data(clustered_data, 'species_data_analyzed.csv')
        
        # Step 3: Feature Selection
        print("\nStep 3: Feature Selection")
        
        # Select features for classification
        feature_selection_clf = select_features(
            preprocessed_data_clf['X_train'], 
            preprocessed_data_clf['y_train'],
            k=3,  # Select top 3 features
            method='k_best',
            is_classification=True
        )
        
        # Select features for regression
        feature_selection_reg = select_features(
            preprocessed_data_reg['X_train'], 
            preprocessed_data_reg['y_train'],
            k=3,  # Select top 3 features
            method='k_best',
            is_classification=False
        )
        
        # Get selected feature names
        selected_features_clf = [
            preprocessed_data_clf['feature_names'][i] 
            for i in feature_selection_clf['selected_indices']
        ]
        
        selected_features_reg = [
            preprocessed_data_reg['feature_names'][i] 
            for i in feature_selection_reg['selected_indices']
        ]
        
        print(f"\nTop classification features: {selected_features_clf}")
        print(f"Top regression features: {selected_features_reg}")
        
        # Step 4: Model Building and Training
        print("\nStep 4: Model Building and Training")
        
        # Classification models
        classification_models = {}
        
        # Train Random Forest Classifier
        rf_classifier = build_model(model_type="random_forest_classifier")
        rf_classifier = train_model(
            rf_classifier,
            preprocessed_data_clf['X_train'],
            preprocessed_data_clf['y_train'],
            preprocessed_data_clf['X_test'],
            preprocessed_data_clf['y_test']
        )
        classification_models["Random Forest"] = rf_classifier
        
        # Train Gradient Boosting Classifier
        gb_classifier = build_model(model_type="gradient_boosting_classifier")
        gb_classifier = train_model(
            gb_classifier,
            preprocessed_data_clf['X_train'],
            preprocessed_data_clf['y_train'],
            preprocessed_data_clf['X_test'],
            preprocessed_data_clf['y_test']
        )
        classification_models["Gradient Boosting"] = gb_classifier
        
        # Train SVM Classifier
        svm_classifier = build_model(model_type="svm_classifier")
        svm_classifier = train_model(
            svm_classifier,
            preprocessed_data_clf['X_train'],
            preprocessed_data_clf['y_train'],
            preprocessed_data_clf['X_test'],
            preprocessed_data_clf['y_test']
        )
        classification_models["SVM"] = svm_classifier
        
        # Regression models
        regression_models = {}
        
        # Train Random Forest Regressor
        rf_regressor = build_model(model_type="random_forest_regressor")
        rf_regressor = train_model(
            rf_regressor,
            preprocessed_data_reg['X_train'],
            preprocessed_data_reg['y_train'],
            preprocessed_data_reg['X_test'],
            preprocessed_data_reg['y_test']
        )
        regression_models["Random Forest"] = rf_regressor
        
        # Train Gradient Boosting Regressor
        gb_regressor = build_model(model_type="gradient_boosting_regressor")
        gb_regressor = train_model(
            gb_regressor,
            preprocessed_data_reg['X_train'],
            preprocessed_data_reg['y_train'],
            preprocessed_data_reg['X_test'],
            preprocessed_data_reg['y_test']
        )
        regression_models["Gradient Boosting"] = gb_regressor
        
        # Train Linear Regression
        linear_regressor = build_model(model_type="linear_regression")
        linear_regressor = train_model(
            linear_regressor,
            preprocessed_data_reg['X_train'],
            preprocessed_data_reg['y_train'],
            preprocessed_data_reg['X_test'],
            preprocessed_data_reg['y_test']
        )
        regression_models["Linear Regression"] = linear_regressor
        
        # Step 5: Cross-Validation
        print("\nStep 5: Cross-Validation")
        
        # Perform cross-validation on the best classification model
        cv_results_clf = perform_cross_validation(
            rf_classifier,
            preprocessed_data_clf['X_train'],
            preprocessed_data_clf['y_train'],
            cv=5,
            scoring='accuracy'
        )
        
        # Perform cross-validation on the best regression model
        cv_results_reg = perform_cross_validation(
            rf_regressor,
            preprocessed_data_reg['X_train'],
            preprocessed_data_reg['y_train'],
            cv=5,
            scoring='r2'
        )
        
        # Step 6: Model Evaluation and Comparison
        print("\nStep 6: Model Evaluation and Comparison")
        
        # Compare classification models
        clf_comparison = compare_models(
            classification_models,
            preprocessed_data_clf['X_test'],
            preprocessed_data_clf['y_test'],
            is_classification=True
        )
        
        # Compare regression models
        reg_comparison = compare_models(
            regression_models,
            preprocessed_data_reg['X_test'],
            preprocessed_data_reg['y_test'],
            is_classification=False
        )
        
        # Evaluate the best classification model
        print("\nDetailed evaluation of the best classification model:")
        best_clf_name = max(clf_comparison, key=lambda k: clf_comparison[k]['accuracy'])
        best_clf = classification_models[best_clf_name]
        
        evaluation_results_clf = evaluate_model(
            best_clf,
            preprocessed_data_clf['X_test'],
            preprocessed_data_clf['y_test'],
            preprocessed_data_clf['status_map'],
            preprocessed_data_clf['feature_names']
        )
        
        # Evaluate the best regression model
        print("\nDetailed evaluation of the best regression model:")
        best_reg_name = max(reg_comparison, key=lambda k: reg_comparison[k]['r2'])
        best_reg = regression_models[best_reg_name]
        
        evaluation_results_reg = evaluate_model(
            best_reg,
            preprocessed_data_reg['X_test'],
            preprocessed_data_reg['y_test'],
            feature_names=preprocessed_data_reg['feature_names']
        )
        
        # Step 7: Generate Conservation Priority Map
        print("\nStep 7: Generating Conservation Priority Map")
        priority_map = plot_conservation_priority_map(
            clustered_data,
            x_feature='habitat_loss',
            y_feature='population_size'
        )
        
        # Save the priority map
        plots_dir = os.path.join(os.path.dirname(__file__), 'data', 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        priority_map.savefig(os.path.join(plots_dir, 'conservation_priority_map.png'))
        plt.close(priority_map)
        
        # Step 8: Generate Conservation Report
        print("\nStep 8: Generating Conservation Report")
        report_path = generate_conservation_report(
            clustered_data,
            output_dir=os.path.join(os.path.dirname(__file__), 'reports')
        )
        
        # Step 9: Generate Species Profiles for Top Priority Species
        print("\nStep 9: Generating Species Profiles")
        profiles_dir = os.path.join(os.path.dirname(__file__), 'profiles')
        
        # Get top 5 priority species
        top_priority_species = priority_species.head(5)
        
        for _, species in top_priority_species.iterrows():
            profile_path = create_species_profile(
                species,
                output_dir=profiles_dir
            )
        
        # Step 10: Save Models
        print("\nStep 10: Saving Models")
        
        # Save the best models
        save_model(best_clf, f'best_classifier_{best_clf_name}.pkl')
        save_model(best_reg, f'best_regressor_{best_reg_name}.pkl')
        
        # Save scalers
        # Extract scalers from preprocessors
        clf_preprocessor = preprocessed_data_clf['preprocessor']
        reg_preprocessor = preprocessed_data_reg['preprocessor']
        # The scaler is in the first transformer of the ColumnTransformer
        clf_scaler = clf_preprocessor.named_transformers_['num'].named_steps['scaler']
        reg_scaler = reg_preprocessor.named_transformers_['num'].named_steps['scaler']
        # Save the scalers
        save_model(clf_scaler, 'classifier_scaler.pkl')
        save_model(reg_scaler, 'regressor_scaler.pkl')
        
        print("\n=== Project Execution Completed Successfully ===")
        print(f"Best Classification Model: {best_clf_name}")
        print(f"Best Regression Model: {best_reg_name}")
        print(f"Conservation Report: {report_path}")
        print(f"Species Profiles: {profiles_dir}")
        print(f"Priority Map: {os.path.join(plots_dir, 'conservation_priority_map.png')}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
