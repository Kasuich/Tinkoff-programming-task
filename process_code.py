import io
import os
import operator
import tokenize as tk
import numpy as np
import pandas as pd
import random

tk_number = operator.itemgetter(0)

class CodeMetrics:
    def __init__(self, num_of_lines, logical_lines, source_lines, comments, multi_comments, blank, single_comments):
        self.num_of_lines = num_of_lines
        self.logical_lines = logical_lines
        self.source_lines = source_lines
        self.comments = comments
        self.multi_comments = multi_comments
        self.blank = blank
        self.single_comments = single_comments

def tokenize_code(code):
    tokenized_code = list(tk.generate_tokens(io.StringIO(code).readline))
    return tokenized_code

def remove_tokens(tokens, rm):
    for i in tokens:
        if i[0] in rm:
            continue
        yield i

def find_token(tokens, token, value):
    for idx, token_values in enumerate(reversed(tokens)):
        if (token, value) == token_values[:2]:
            return len(tokens) - idx - 1
    return ValueError('token with value not found')

def split_tks(tokens,token, value):
    res = [[]]
    for vals in tokens:
        if (token, value) == vals[:2]:
            res.append([])
            continue
        res[-1].append(vals)
    return res

def process_line_w_all_tokens(line, lines):
    buf = line
    used_lines = [line]
    while True:
        try:
            tokens = tokenize_code(buf)
        except tk.TokenError:
            pass
        else:
            if not any(tok[0] == tk.ERRORTOKEN for tok in tokens):
                return tokens, used_lines

        next_line = next(lines)
        buf = buf + '\n' + next_line
        used_lines.append(next_line)


def get_num_of_logical(all_tokens):
    def tmp(tokens):
        computed = list(remove_tokens(tokens, [tk.COMMENT, tk.NL, tk.NEWLINE]))
        try:
            pos = find_token(computed, tk.OP, ':')
            return 2 - (pos == len(computed) - 2)
        except ValueError:
            if not list(remove_tokens(computed, [tk.NL, tk.NEWLINE, tk.ENDMARKER])):
                return 0
            return 1
    return sum(tmp(k) for k in split_tks(all_tokens, tk.OP, ';'))

def if_single(token_number, tokens):
    return tk_number(tokens[0]) == token_number and all(
        tk_number(t) in (tk.ENDMARKER, tk.NL, tk.NEWLINE) for t in tokens[1:]
    )

def count_metrics(source):
    logical_lines = comments = single_comments = multi_comments = blank = source_lines = 0
    lines = (l.strip() for l in source.splitlines())
    tmpo = 1
    for line in lines:
        try:
            tokens, parsed_lines = process_line_w_all_tokens(line, lines)
        except:
            raise SyntaxError('SyntaxError')
        
        tmpo += len(parsed_lines)
        
        comments += sum(1 for t in tokens if tk_number(t) == tk.COMMENT)
        if if_single(tk.COMMENT, tokens):
            single_comments += 1
            
        elif if_single(tk.STRING, tokens):
            _, _, (start_row, _), (end_row, _), _ = tokens[0]
            if end_row == start_row:
                single_comments += 1
            else:
                multi_comments += sum(1 for l in parsed_lines if l)
                blank += sum(1 for l in parsed_lines if not l)
        
        else:
            for parsed_line in parsed_lines:
                if parsed_line:
                    source_lines += 1
                else:
                    blank += 1
                    
        logical_lines += get_num_of_logical(tokens)
    
    num_of_lines = source_lines + blank + multi_comments + single_comments
    return CodeMetrics(num_of_lines, logical_lines, source_lines, comments, multi_comments, blank, single_comments)
        
def levenstein(a, b):
    size_a = len(a) + 1
    size_b = len(b) + 1
    matrix = np.zeros((size_a, size_b))
    for x in np.arange(size_a):
        matrix[x, 0] = x
    for y in np.arange(size_b):
        matrix[0, y] = y
    for x in np.arange(1, size_a):
        for y in np.arange(1, size_b):
            if a[x - 1] == b[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return int(matrix[size_a - 1, size_b - 1])

def process_two_files(code1, code2, is_plagiat=None, use_levenstein=False):
    
    metrics1 = count_metrics(code1)
    metrics2 = count_metrics(code2)
    
    res = pd.DataFrame.from_dict({
        'num_of_lines': [metrics1.num_of_lines - metrics2.num_of_lines],
        'logical_lines': [metrics1.logical_lines - metrics2.logical_lines],
        'source_lines': [metrics1.source_lines - metrics2.source_lines],
        'comments': [metrics1.comments - metrics2.comments],
        'multi_comments': [metrics1.multi_comments - metrics2.multi_comments],
        'blank': [metrics1.blank - metrics2.blank],
        'single_comments': [metrics1.single_comments - metrics2.single_comments],
    })
    
    if use_levenstein:
        res['levenstein'] = levenstein(code1, code2)
        
    if is_plagiat != None:
        res['is_plagiat'] = is_plagiat
    
    return res.apply(abs)

def process_two_directories(dir1, dir2, random_shuffle=False):
    
    codes1 = []
    codes2 = []
    
    res = pd.DataFrame()
    
    for root, dirs, files in os.walk(dir1, topdown=False):
        for name in files:
            with io.open(os.path.join(dir1, name), encoding='utf-8') as f:
                codes1.append(f.read())
                
    for root, dirs, files in os.walk(dir2, topdown=False):
        for name in files:
            with io.open(os.path.join(dir2, name), encoding='utf-8') as f:
                codes2.append(f.read())   
                
    for i in range(len(codes1)):
        res = pd.concat([res, process_two_files(codes1[i], codes2[i], is_plagiat=1)], ignore_index=True)
        
    if random_shuffle:
        
        for i in range(len(codes1)):
            
            rand1 = random.randrange(0, len(codes1))
            rand2 = random.randrange(0, len(codes2))
            
            res1 = process_two_files(codes1[rand1], codes2[rand2], is_plagiat=int(rand1 == rand2))
            
            res = pd.concat([res, res1], ignore_index=True)
        
    return res
