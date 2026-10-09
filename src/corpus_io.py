"""Original LCCC parsing/normalization; no corpus is redistributed."""
import hashlib,json,re,unicodedata
LABELS=['吗', '呢', '吧', '嘛', '呗', '罢了', '啊', '哦', '呀', '啦', '么', '咯']
SORTED_LABELS=sorted(LABELS,key=len,reverse=True)
END_PUNCT='。！？!?.,，；;：:…~～、"\'”’）)]】》〉」』 '

def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def normalize(s):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', s)).lower()

def detokenize(s):
    s = re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', s.strip())
    s = re.sub(r'\s+([，。！？；：、,.!?;:])', r'\1', s)
    return re.sub(r'([，。！？；：、,.!?;:])\s+', r'\1', s)

def extract(turn):
    text = detokenize(turn)
    core = text.rstrip(END_PUNCT)
    label = next((p for p in SORTED_LABELS if core.endswith(p)), None)
    if label is None:
        return None, 'no_target_suffix'
    if label == '么' and core.endswith(('什么', '怎么', '这么', '那么', '多么', '要么')):
        return None, 'lexical_me'
    if label == '嘛' and core.endswith('干嘛'):
        return None, 'lexical_ganma'
    if label == '罢了' and core.endswith('作罢了'):
        return None, 'lexical_bale'
    body = core[:-len(label)].rstrip()
    if not 6 <= len(normalize(body)) <= 100:
        return None, 'body_length'
    if len(re.findall(r'[\u3400-\u9fff]', body)) < 4:
        return None, 'insufficient_chinese'
    if re.search(r'https?://|www\.|@[\w\u3400-\u9fff]|1[3-9]\d{9}', text):
        return None, 'url_mention_or_phone'
    if any(body.endswith(p) for p in SORTED_LABELS):
        return None, 'particle_stack'
    return {'target_body': body, 'target_punctuation': text[len(core):],
            'observed_particle': label, 'target_original': text}, None

def stream_array(stream):
    """Decode one top-level JSON array incrementally with bounded buffers."""
    decoder, buf, pos, eof, begun = json.JSONDecoder(), '', 0, False, False
    while True:
        if pos > 524288:
            buf, pos = buf[pos:], 0
        while pos < len(buf) and buf[pos] in ' \r\n\t,':
            pos += 1
        if not begun and pos < len(buf):
            if buf[pos] != '[': raise ValueError('expected top-level array')
            pos += 1
            begun = True
            continue
        if begun and pos < len(buf) and buf[pos] == ']':
            if buf[pos + 1:].strip() or stream.read().strip():
                raise ValueError('trailing data')
            return
        if begun and pos < len(buf):
            try:
                value, end = decoder.raw_decode(buf, pos)
                pos = end
                yield value
                continue
            except json.JSONDecodeError:
                if eof: raise
        if eof: raise ValueError('unexpected end of JSON array')
        chunk = stream.read(1048576)
        if not chunk: eof = True
        buf = buf[pos:] + chunk
        pos = 0
