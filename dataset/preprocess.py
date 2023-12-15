from io import StringIO
import tokenize
import jsonlines
import re

import json
import os

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)



if __name__ == "__main__":
    for language in ['javascript', 'python', 'java']:
        print(language)
        train, valid, test = [], [], []
        for root, dirs, files in os.walk(language + '/final'):
            for file in files:
                temp = os.path.join(root, file)
                if '.jsonl' in temp:
                    if 'train' in temp:
                        train.append(temp)
                    elif 'valid' in temp:
                        valid.append(temp)
                    elif 'test' in temp:
                        test.append(temp)

        train_data, valid_data, test_data = {}, {}, {}
        for files, data in [[train, train_data], [valid, valid_data], [test, test_data]]:
            for file in files:
                if '.gz' in file:
                    os.system("gzip -d {}".format(file))
                    file = file.replace('.gz', '')
                with open(file) as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        data[js['url']] = js
        for tag, data in [['train', train_data], ['valid', valid_data], ['test', test_data]]:
            with open('{}/{}.jsonl'.format(language, tag), 'w') as f, open("{}/{}.txt".format(language, tag)) as f1:
                for line in f1:
                    line = line.strip()
                    if line in data:
                        f.write(json.dumps(data[line]) + '\n')


        data_type = ['train', 'valid', 'test']

        for mode in data_type:
            data = []
            idx = 0
            with open("./{}/{}.jsonl".format(language, mode), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)

                    try:
                        code = js['code']
                    except:
                        code = js['function']

                    try:
                        clean_code = remove_comments_and_docstrings(code, 'java')
                    except:
                        clean_code = code
                        idx = idx + 1

                    js['clean_code'] = clean_code
                    js['clean_doc'] = " ".join(js["docstring_tokens"])
                    data.append(js)

            print("./{}/clean_{}.jsonl  {}".format(language, mode, len(data)))
            with jsonlines.open("./{}/clean_{}.jsonl".format(language, mode), 'w') as w:
                for line in data:
                    w.write(line)

