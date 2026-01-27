from django.core.management.base import BaseCommand
from exams.models import Problem, TestCase, Language, ProblemLanguage


class Command(BaseCommand):
    help = 'Populate database with sample coding problems and test cases'

    def handle(self, *args, **options):
        self.stdout.write('Creating languages...')
        
        # Create languages
        js_lang, created = Language.objects.get_or_create(
            name='javascript',
            defaults={
                'display_name': 'JavaScript',
                'docker_image': 'node:latest',
                'execute_command': 'node',
                'file_extension': '.js',
                'default_code': 'function solve(...args) {\n  return null;\n}'
            }
        )
        if created:
            self.stdout.write(f'Created language: {js_lang.display_name}')
        
        python_lang, created = Language.objects.get_or_create(
            name='python',
            defaults={
                'display_name': 'Python',
                'docker_image': 'python:latest',
                'execute_command': 'python',
                'file_extension': '.py',
                'default_code': 'def solve(*args):\n    return None'
            }
        )
        if created:
            self.stdout.write(f'Created language: {python_lang.display_name}')
        
        self.stdout.write('Creating sample problems...')
        
        # Problem 1: Sum of two numbers
        problem1, created = Problem.objects.get_or_create(
            title="Sum of two numbers",
            defaults={
                'description': 'Implement a function <code>solve(a, b)</code> that returns the sum of its two numeric arguments.',
                'problem_statement': 'Write a function that takes two numbers as input and returns their sum.',
                'input_format': 'Two numbers a and b',
                'output_format': 'Return the sum of a and b',
                'constraints': 'a and b are integers between -1000 and 1000',
                'time_limit_seconds': 5,
                'points': 100
            }
        )
        
        if created:
            self.stdout.write(f'Created problem: {problem1.title}')
            
            # Test cases for problem 1
            test_cases_1 = [
                {'name': 'Small positive numbers', 'input_data': [1, 2], 'expected_output': 3, 'order': 1},
                {'name': 'Zero values', 'input_data': [0, 0], 'expected_output': 0, 'order': 2},
                {'name': 'With a negative', 'input_data': [-5, 3], 'expected_output': -2, 'order': 3},
                {'name': 'Large integers', 'input_data': [1000000, 2345678], 'expected_output': 3345678, 'order': 4},
                {'name': 'Floating point', 'input_data': [1.5, 2.25], 'expected_output': 3.75, 'order': 5}
            ]
            
            for tc_data in test_cases_1:
                TestCase.objects.create(problem=problem1, **tc_data)
            
            # Create language-specific starter code
            ProblemLanguage.objects.get_or_create(
                problem=problem1,
                language=js_lang,
                defaults={
                    'starter_code': 'function solve(a, b) {\n  // Return the sum of two numbers\n  return a + b;\n}'
                }
            )
            
            ProblemLanguage.objects.get_or_create(
                problem=problem1,
                language=python_lang,
                defaults={
                    'starter_code': 'def solve(a, b):\n    # Return the sum of two numbers\n    return a + b'
                }
            )
            
            self.stdout.write(f'Created {len(test_cases_1)} test cases for {problem1.title}')
        else:
            self.stdout.write(f'Problem already exists: {problem1.title}')
        
        # Problem 2: Multiply two numbers
        problem2, created = Problem.objects.get_or_create(
            title="Multiply two numbers",
            defaults={
                'description': 'Implement a function <code>solve(a, b)</code> that returns the product of its two numeric arguments.',
                'problem_statement': 'Write a function that takes two numbers as input and returns their product.',
                'input_format': 'Two numbers a and b',
                'output_format': 'Return the product of a and b',
                'constraints': 'a and b are integers between -1000 and 1000',
                'time_limit_seconds': 5,
                'points': 100
            }
        )
        
        if created:
            self.stdout.write(f'Created problem: {problem2.title}')
            
            # Test cases for problem 2
            test_cases_2 = [
                {'name': 'Small positive numbers', 'input_data': [2, 3], 'expected_output': 6, 'order': 1},
                {'name': 'Zero values', 'input_data': [0, 5], 'expected_output': 0, 'order': 2},
                {'name': 'With a negative', 'input_data': [-4, 3], 'expected_output': -12, 'order': 3},
                {'name': 'Large integers', 'input_data': [1000, 2345], 'expected_output': 2345000, 'order': 4},
                {'name': 'Floating point', 'input_data': [1.5, 2], 'expected_output': 3, 'order': 5}
            ]
            
            for tc_data in test_cases_2:
                TestCase.objects.create(problem=problem2, **tc_data)
            
            # Create language-specific starter code
            ProblemLanguage.objects.get_or_create(
                problem=problem2,
                language=js_lang,
                defaults={
                    'starter_code': 'function solve(a, b) {\n  // Return the product of two numbers\n  return a * b;\n}'
                }
            )
            
            ProblemLanguage.objects.get_or_create(
                problem=problem2,
                language=python_lang,
                defaults={
                    'starter_code': 'def solve(a, b):\n    # Return the product of two numbers\n    return a * b'
                }
            )
            
            self.stdout.write(f'Created {len(test_cases_2)} test cases for {problem2.title}')
        else:
            self.stdout.write(f'Problem already exists: {problem2.title}')
        
        # Problem 3: Find maximum of two numbers
        problem3, created = Problem.objects.get_or_create(
            title="Find maximum of two numbers",
            defaults={
                'description': 'Implement a function <code>solve(a, b)</code> that returns the maximum of its two numeric arguments.',
                'problem_statement': 'Write a function that takes two numbers as input and returns the maximum of the two.',
                'input_format': 'Two numbers a and b',
                'output_format': 'Return the maximum of a and b',
                'constraints': 'a and b are integers between -1000 and 1000',
                'time_limit_seconds': 5,
                'points': 100
            }
        )
        
        if created:
            self.stdout.write(f'Created problem: {problem3.title}')
            
            # Test cases for problem 3
            test_cases_3 = [
                {'name': 'Positive numbers', 'input_data': [5, 3], 'expected_output': 5, 'order': 1},
                {'name': 'Negative numbers', 'input_data': [-10, -5], 'expected_output': -5, 'order': 2},
                {'name': 'Equal numbers', 'input_data': [7, 7], 'expected_output': 7, 'order': 3},
                {'name': 'Mixed signs', 'input_data': [-3, 8], 'expected_output': 8, 'order': 4},
                {'name': 'Zero and positive', 'input_data': [0, 12], 'expected_output': 12, 'order': 5}
            ]
            
            for tc_data in test_cases_3:
                TestCase.objects.create(problem=problem3, **tc_data)
            
            # Create language-specific starter code
            ProblemLanguage.objects.get_or_create(
                problem=problem3,
                language=js_lang,
                defaults={
                    'starter_code': 'function solve(a, b) {\n  // Return the maximum of two numbers\n  return Math.max(a, b);\n}'
                }
            )
            
            ProblemLanguage.objects.get_or_create(
                problem=problem3,
                language=python_lang,
                defaults={
                    'starter_code': 'def solve(a, b):\n    # Return the maximum of two numbers\n    return max(a, b)'
                }
            )
            
            self.stdout.write(f'Created {len(test_cases_3)} test cases for {problem3.title}')
        else:
            self.stdout.write(f'Problem already exists: {problem3.title}')
        
        self.stdout.write(self.style.SUCCESS('Successfully populated database with sample problems!')) 