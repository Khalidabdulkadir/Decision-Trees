def factorial(n, depth=0):
    # Print the current call with indentation based on recursion depth.
    print("  " * depth + f"factorial({n}) called")
    
    # Base case: when n equals 1.
    if n == 1:
        print("  " * depth + f"Base case reached: factorial(1) = 1")
        return 1
    else:
        # Recursive case: call factorial with n-1.
        result = n * factorial(n - 1, depth + 1)
        print("  " * depth + f"Returning from factorial({n}): {result}")
        return result

# Call the function and print the final result.
final_result = factorial(5)
print(f"\nFinal Result: factorial(5) = {final_result}")
