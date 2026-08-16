import re
from typing import Union

class Calculator:
    """A simple calculator tool for evaluating basic arithmetic expressions."""
    
    @staticmethod
    def evaluate_expression(expression: str) -> Union[float, str]:
        """Evaluate a basic arithmetic expression.
        
        Supports only basic arithmetic operations (+, -, *, /) and parentheses.
        Returns an error message if the expression is invalid or cannot be 
        evaluated safely.
        
        Args:
            expression: A string containing a mathematical expression
                       e.g. "5 + 3" or "10 * (2 + 3)"
            
        Returns:
            Union[float, str]: The result of the evaluation, or an error message
                              if the expression is invalid
        
        Examples:
            >>> Calculator.evaluate_expression("5 + 3")
            8.0
            >>> Calculator.evaluate_expression("10 * (2 + 3)")
            50.0
            >>> Calculator.evaluate_expression("15 / 3")
            5.0
        """
        try:
            # Clean up the expression
            expression = expression.strip()
            
            # Only allow safe characters (digits, basic operators, parentheses, spaces)
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.]*$', expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}})
            
            # Convert to float and handle division by zero
            return float(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (SyntaxError, TypeError, NameError):
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {str(e)}"


    def evaluate_expression_percentage(self, expression: str) -> Union[float, str]:
        """Evaluate an arithmetic expression that may include percentages.
        
        This method converts percentage values into their decimal equivalents
        before evaluating the expression. For example, "50%" becomes 0.5.
        
        Args:
            expression: A string containing a mathematical expression
                       e.g. "50% + 20" or "10 * (2 + 30%)"
            
        Returns:
            Union[float, str]: The result of the evaluation, or an error message
                              if the expression is invalid
        
        Examples:
            >>> Calculator().evaluate_expression_percentage("50% + 20")
            20.5
            >>> Calculator().evaluate_expression_percentage("10 * (2 + 30%)")
            22.0
        """
        # try:
        #     # Replace percentage values with their decimal equivalents
        #     expression = re.sub(r'(\d+(\.\d+)?)\s*%', lambda m: str(float(m.group(1)) / 100), expression)
            
        #     # Evaluate the modified expression using the existing method
        #     return self.evaluate_expression(expression)
            
        # except Exception as e:
        #     return f"Error: {str(e)}"

        try:
            # Clean up the expression
            expression = expression.strip()

            # Replace percentage values: "50%" → "(50/100)"
            expression = re.sub(r'(\d+(\.\d+)?)%', r'(\1/100)', expression)

            # Only allow safe characters (digits, operators, parentheses, spaces, dots, slashes)
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.]*$', expression):
                return "Error: Invalid characters in expression"

            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}})

            return float(result)

        except ZeroDivisionError:
            return "Error: Division by zero"
        except (SyntaxError, TypeError, NameError):
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {str(e)}"
