#!/bin/bash

echo "================================================================"
echo "Minibatch GW Transport Matrix Computation (LOG SCALE)"
echo "================================================================"
echo ""
echo "This script will:"
echo "  1. Compute 500 random 32-cell minibatch transport matrices"
echo "  2. Save matrices in LOG SCALE format"
echo "  3. Analyze the results"
echo ""
echo "Output directory: /home/users/turbodu/.../Transport_Matrix/"
echo ""
echo "================================================================"
echo ""

# Step 1: Compute transport matrices
echo "Step 1: Computing Transport Matrices"
echo "--------------------------------------"
echo "This will take approximately 10-20 minutes..."
echo ""

python compute_minibatch_gw_logscale.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Transport matrix computation completed successfully!"
    echo ""
    
    # Step 2: Analyze results
    echo "================================================================"
    echo "Step 2: Analyzing Transport Matrices"
    echo "--------------------------------------"
    echo ""
    
    python analyze_minibatch_gw_logscale.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Analysis completed successfully!"
        echo ""
        echo "================================================================"
        echo "✓ All Done!"
        echo "================================================================"
        echo ""
        echo "Generated files:"
        echo "  - 500 transport matrix CSV files (32×32, LOG SCALE)"
        echo "  - computation_summary.txt"
        echo "  - verification_statistics.csv"
        echo "  - batch_statistics_logscale.csv"
        echo "  - transport_matrices_analysis_logscale.png"
        echo "  - analysis_summary_logscale.txt"
        echo ""
        echo "Location:"
        echo "  /home/users/turbodu/.../Transport_Matrix/"
        echo ""
    else
        echo "✗ Analysis failed. Check error messages above."
        exit 1
    fi
else
    echo "✗ Transport matrix computation failed. Check error messages above."
    exit 1
fi