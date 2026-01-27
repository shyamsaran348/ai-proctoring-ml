# exams/services/code_executor.py - Secure Code Execution Service

import os
import tempfile
import subprocess
import json
import time
import hashlib
import docker
from typing import Dict, List, Any, Tuple
from django.conf import settings
from django.utils import timezone
from ..models import Language, TestCase, Submission


class CodeExecutionService:
    """Secure code execution service using Docker containers"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.temp_dir = getattr(settings, 'CODE_EXECUTION_TEMP_DIR', '/tmp/code_execution')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def execute_submission(self, submission: Submission, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Execute a submission against test cases"""
        language = submission.language
        code = submission.code
        
        # Prepare execution environment
        execution_id = self.generate_execution_id()
        work_dir = os.path.join(self.temp_dir, execution_id)
        os.makedirs(work_dir, exist_ok=True)
        
        try:
            # Write code to file
            code_file = self.write_code_to_file(code, language, work_dir)
            
            # Compile if necessary
            compile_result = self.compile_code(code_file, language, work_dir)
            if not compile_result['success']:
                return {
                    'status': 'compilation_error',
                    'error_message': compile_result['error'],
                    'test_results': []
                }
            
            # Execute against test cases
            test_results = []
            total_score = 0
            
            for test_case in test_cases:
                result = self.run_test_case(
                    code_file, language, test_case, work_dir, execution_id
                )
                test_results.append(result)
                
                if result['status'] == 'accepted':
                    total_score += test_case.points
            
            # Determine overall status
            if all(r['status'] == 'accepted' for r in test_results):
                status = 'accepted'
            elif any(r['status'] == 'time_limit_exceeded' for r in test_results):
                status = 'time_limit_exceeded'
            elif any(r['status'] == 'memory_limit_exceeded' for r in test_results):
                status = 'memory_limit_exceeded'
            elif any(r['status'] == 'runtime_error' for r in test_results):
                status = 'runtime_error'
            else:
                status = 'wrong_answer'
            
            return {
                'status': status,
                'score': total_score,
                'test_results': test_results,
                'execution_time': max(r.get('execution_time', 0) for r in test_results),
                'memory_used': max(r.get('memory_used', 0) for r in test_results)
            }
        
        finally:
            # Cleanup
            self.cleanup_execution_directory(work_dir)
    
    def run_test_case(self, code_file: str, language: Language, test_case: TestCase, 
                     work_dir: str, execution_id: str) -> Dict[str, Any]:
        """Run code against a single test case"""
        
        # Prepare input
        input_file = os.path.join(work_dir, f'input_{test_case.id}.txt')
        with open(input_file, 'w') as f:
            f.write(test_case.input_data)
        
        # Prepare Docker command
        docker_command = self.build_docker_command(
            language, code_file, input_file, work_dir, execution_id
        )
        
        start_time = time.time()
        
        try:
            # Run in Docker container
            result = self.docker_client.containers.run(
                image=language.docker_image,
                command=docker_command,
                volumes={
                    work_dir: {'bind': '/workspace', 'mode': 'rw'}
                },
                working_dir='/workspace',
                mem_limit=f'{language.max_memory_mb}m',
                memswap_limit=f'{language.max_memory_mb}m',
                cpu_quota=language.max_cpu_usage * 1000,
                cpu_period=100000,
                network_disabled=True,
                remove=True,
                timeout=language.max_execution_time,
                stdout=True,
                stderr=True
            )
            
            execution_time = time.time() - start_time
            output = result.decode('utf-8').strip()
            
            # Compare output
            expected_output = test_case.expected_output.strip()
            is_correct = self.compare_outputs(output, expected_output)
            
            return {
                'test_case_id': test_case.id,
                'test_case_name': test_case.name,
                'status': 'accepted' if is_correct else 'wrong_answer',
                'input': test_case.input_data,
                'expected_output': expected_output,
                'actual_output': output,
                'execution_time': execution_time,
                'memory_used': 0,  # Docker doesn't easily provide this
                'is_correct': is_correct
            }
            
        except docker.errors.ContainerError as e:
            execution_time = time.time() - start_time
            return {
                'test_case_id': test_case.id,
                'test_case_name': test_case.name,
                'status': 'runtime_error',
                'input': test_case.input_data,
                'expected_output': test_case.expected_output.strip(),
                'actual_output': '',
                'execution_time': execution_time,
                'memory_used': 0,
                'error_message': str(e),
                'is_correct': False
            }
            
        except Exception as e:
            if 'timeout' in str(e).lower():
                status = 'time_limit_exceeded'
            else:
                status = 'runtime_error'
            
            return {
                'test_case_id': test_case.id,
                'test_case_name': test_case.name,
                'status': status,
                'input': test_case.input_data,
                'expected_output': test_case.expected_output.strip(),
                'actual_output': '',
                'execution_time': language.max_execution_time,
                'memory_used': 0,
                'error_message': str(e),
                'is_correct': False
            }
    
    def build_docker_command(self, language: Language, code_file: str, 
                           input_file: str, work_dir: str, execution_id: str) -> List[str]:
        """Build Docker execution command for specific language"""
        
        filename = os.path.basename(code_file)
        input_filename = os.path.basename(input_file)
        
        if language.name == 'python3':
            return ['python3', filename, '<', input_filename]
        
        elif language.name == 'java':
            class_name = filename.replace('.java', '')
            return ['sh', '-c', f'javac {filename} && java {class_name} < {input_filename}']
        
        elif language.name == 'cpp':
            executable = filename.replace('.cpp', '')
            return ['sh', '-c', f'g++ -o {executable} {filename} && ./{executable} < {input_filename}']
        
        elif language.name == 'javascript':
            return ['node', filename, '<', input_filename]
        
        else:
            # Default execution
            return ['sh', '-c', f'{language.execute_command.format(filename=filename)} < {input_filename}']
    
    def write_code_to_file(self, code: str, language: Language, work_dir: str) -> str:
        """Write code to appropriate file"""
        
        if language.name == 'java':
            # Extract class name from Java code
            class_name = self.extract_java_class_name(code)
            filename = f'{class_name}.java'
        else:
            filename = f'solution{language.file_extension}'
        
        file_path = os.path.join(work_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return file_path
    
    def extract_java_class_name(self, code: str) -> str:
        """Extract public class name from Java code"""
        import re
        match = re.search(r'public\s+class\s+(\w+)', code)
        return match.group(1) if match else 'Solution'
    
    def compile_code(self, code_file: str, language: Language, work_dir: str) -> Dict[str, Any]:
        """Compile code if compilation is required"""
        
        if not language.compile_command:
            return {'success': True}
        
        try:
            filename = os.path.basename(code_file)
            
            if language.name == 'java':
                compile_cmd = ['javac', filename]
            elif language.name == 'cpp':
                executable = filename.replace('.cpp', '')
                compile_cmd = ['g++', '-o', executable, filename]
            else:
                compile_cmd = language.compile_command.format(filename=filename).split()
            
            result = self.docker_client.containers.run(
                image=language.docker_image,
                command=compile_cmd,
                volumes={
                    work_dir: {'bind': '/workspace', 'mode': 'rw'}
                },
                working_dir='/workspace',
                remove=True,
                timeout=30,  # Compilation timeout
                stdout=True,
                stderr=True
            )
            
            return {'success': True}
            
        except docker.errors.ContainerError as e:
            error_output = e.stderr.decode('utf-8') if e.stderr else str(e)
            return {
                'success': False,
                'error': error_output
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_outputs(self, actual: str, expected: str) -> bool:
        """Compare actual and expected outputs"""
        
        # Normalize whitespace
        actual_lines = [line.strip() for line in actual.strip().split('\n')]
        expected_lines = [line.strip() for line in expected.strip().split('\n')]
        
        # Remove empty lines at the end
        while actual_lines and not actual_lines[-1]:
            actual_lines.pop()
        while expected_lines and not expected_lines[-1]:
            expected_lines.pop()
        
        return actual_lines == expected_lines
    
    def generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
    
    def cleanup_execution_directory(self, work_dir: str):
        """Clean up execution directory"""
        try:
            import shutil
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


class CodeExecutionView:
    """Enhanced code execution view with multi-language support"""
    
    def __init__(self):
        self.executor = CodeExecutionService()
    
    def execute_code(self, code: str, language_name: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute code with given test cases"""
        
        try:
            # Get language configuration
            language = Language.objects.get(name=language_name, is_active=True)
        except Language.DoesNotExist:
            return {
                'success': False,
                'error': f'Language {language_name} not supported'
            }
        
        # Create temporary submission for execution
        execution_id = self.executor.generate_execution_id()
        work_dir = os.path.join(self.executor.temp_dir, execution_id)
        os.makedirs(work_dir, exist_ok=True)
        
        try:
            # Write code to file
            code_file = self.executor.write_code_to_file(code, language, work_dir)
            
            # Compile if necessary
            compile_result = self.executor.compile_code(code_file, language, work_dir)
            if not compile_result['success']:
                return {
                    'success': False,
                    'error': compile_result['error'],
                    'results': []
                }
            
            # Execute test cases
            results = []
            
            for i, test_case_data in enumerate(test_cases):
                # Create temporary test case object
                class TempTestCase:
                    def __init__(self, data, index):
                        self.id = index
                        self.name = data.get('name', f'Test {index + 1}')
                        self.input_data = data.get('input', '')
                        self.expected_output = data.get('expected', '')
                        self.points = data.get('points', 1)
                
                temp_test_case = TempTestCase(test_case_data, i)
                result = self.executor.run_test_case(
                    code_file, language, temp_test_case, work_dir, execution_id
                )
                results.append(result)
            
            return {
                'success': True,
                'results': results,
                'execution_time': max(r.get('execution_time', 0) for r in results) if results else 0
            }
        
        finally:
            # Cleanup
            self.executor.cleanup_execution_directory(work_dir)


# Language configurations for Docker
LANGUAGE_CONFIGS = [
    {
        'name': 'python3',
        'display_name': 'Python 3',
        'docker_image': 'python:3.9-slim',
        'file_extension': '.py',
        'execute_command': 'python3 {filename}',
        'compile_command': '',
        'default_code': '''def solve():
    # Your code here
    pass

if __name__ == "__main__":
    solve()''',
        'max_execution_time': 5,
        'max_memory_mb': 256
    },
    {
        'name': 'java',
        'display_name': 'Java',
        'docker_image': 'openjdk:11-jdk-slim',
        'file_extension': '.java',
        'execute_command': 'java {classname}',
        'compile_command': 'javac {filename}',
        'default_code': '''import java.util.*;
import java.io.*;

public class Solution {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        // Your code here
    }
}''',
        'max_execution_time': 10,
        'max_memory_mb': 512
    },
    {
        'name': 'cpp',
        'display_name': 'C++',
        'docker_image': 'gcc:9',
        'file_extension': '.cpp',
        'execute_command': './{executable}',
        'compile_command': 'g++ -o {executable} {filename}',
        'default_code': '''#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // Your code here
    return 0;
}''',
        'max_execution_time': 5,
        'max_memory_mb': 256
    },
    {
        'name': 'javascript',
        'display_name': 'JavaScript (Node.js)',
        'docker_image': 'node:16-slim',
        'file_extension': '.js',
        'execute_command': 'node {filename}',
        'compile_command': '',
        'default_code': '''// Your code here
function solve() {
    
}

solve();''',
        'max_execution_time': 5,
        'max_memory_mb': 256
    }
]