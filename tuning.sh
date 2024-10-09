#!/bin/bash

# set lambdas and sigmas
lambdas=(0.1 0.5 1 2)
sigmas=(0.01 0.1 0.5 1)

output_file="grid_search_results.txt"

if [ -f "$output_file" ]; then
    rm "$output_file"
fi

touch "$output_file"

# grid search: iterate over all combinations of lambda and sigma
for lambda in "${lambdas[@]}"; do
    for sigma in "${sigmas[@]}"; do
        echo "Running with lambda=$lambda, sigma=$sigma"
        
        mpiexec -n 8 python main.py -N 8 -L "$lambda" -S "$sigma" &> temp_output.txt

        echo "lambda=$lambda, sigma=$sigma" >> "$output_file"
        cat temp_output.txt >> "$output_file"
        echo -e "\n---------------------------------\n" >> "$output_file"
        sleep 1
    done
done

rm temp_output.txt

echo "Grid search completed. Results saved to $output_file."