#!/usr/bin/env python3

# from https://github.com/ufal/asr_latency

import sys

def eprint(*a,**kw):
    print(*a,**kw,file=sys.stderr)

def continuous_levenshtein_alignment(gt_words, asr_words):
    """
    Aligns two sequences of units called 'word' (which can be characters, actually) using a dynamic programming approach 
    similar to Levenshtein minimum edit distance. 
    The innovative adaptation is that it prioritizes continuous sequence of substitutions and copies over interruptions with insertions and deletions.
    We call this approach ``Continous Levenshtein Alignment'' in CUNI submission to IWSLT 2025 Simultaneous task.
    
    Args:
        gt_words (list): List of dictionaries representing ground truth words with 'word' and 'time'.
        asr_words (list): List of dictionaries representing ASR output words with 'word' and 'time'.

    example ('word' is a character):
        gt_words = asr_words = [{'word': 'H', 'time': 1.113}, {'word': 'e', 'time': 1.113}, ... ]
        
    Returns:
        list: A list of tuples, each containing a pair of aligned words from ground truth and ASR, or None for insertion/deletion.

    exaple return:

        [ ({'word': 'H', 'time': 1.113}, {'word': 'H', 'time': 2.6}),
            ...
          (None, {'word': 'e', 'time': 1.113})
        ]

    """

    
    # Initialize a 2D array to store distances
    m = len(gt_words)
    n = len(asr_words)
    # dp = dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # just ofr an easier access
    gt = [w["word"] for w in gt_words]
    asr = [w["word"] for w in asr_words]

    # Fill the Dynamic Programming table, like in the regular Levenshtein minimum edit distance alignment
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if gt[i-1] == asr[j-1] else 1
                dp[i][j] = min(dp[i][j-1] + 1,      # Insert
                               dp[i-1][j] + 1,      # Delete
                               dp[i-1][j-1] + cost) # Substitute or match



    # for better code readability
    def delete_possible(i,j):
        return dp[i][j] == dp[i-1][j] + 1
    def insert_possible(i,j):
        return dp[i][j] == dp[i][j-1] + 1
    def copy_sub_possible(i,j):
        cost = 0 if gt[i-1] == asr[j-1] else 1
        if dp[i][j] == dp[i-1][j-1] + cost:
            return True
   
    # Backtrack to find the alignment 
    i, j = m, n
    alignment = []

    #### The adaptation for *Continuous* Levenshtein Alignment
    # To find the alignment with most substitutions and matches in the row,
    # we try operations in this order. 
    priorities = ["copy_sub", "delete", "insert"]

    while i > 0 and j > 0:
        for p in priorities:
            if p == "copy_sub" and copy_sub_possible(i,j):
                alignment.append((gt_words[i-1], asr_words[j-1]))
                i -= 1
                j -= 1
                # Whenever the continuation is not possible, we re-order the priorities for the next time.
                new_priorities = ["copy_sub", "delete", "insert"]  # as long as we can copy/sub, we do
                break
            if p == "delete" and delete_possible(i,j):
                alignment.append((gt_words[i-1], None))
                i -= 1
                new_priorities = ["delete", "insert", "copy_sub"]  # otherwise we prefer delete/insert over copy
                break 
            if p == "insert" and insert_possible(i,j):
                alignment.append((None, asr_words[j-1]))
                j -= 1
                new_priorities = ["insert", "delete", "copy_sub"]
                break
        priorities = new_priorities

    # deletions at the end
    while i > 0:
        alignment.append((gt_words[i-1], None))
        i -= 1

    # insertions at the end
    while j > 0:
        alignment.append((None, asr_words[j-1]))
        j -= 1

    # the alignment are ordered bottom-to-top. We reverse them, to be returned in top-to-bottom order
    alignment.reverse()
    return alignment
   

def calculate_latency(alignment):
    """
    Calculates the average latency from an aligned list of word pairs.
    
    Args:
        alignment (list): List of tuples containing ground truth and ASR words.
        
    Returns:
        float: The average latency in seconds.
    """
    total_latency = 0.0
    count = 0
    
    for gt_word, asr_word in alignment:
        # Ensure times are in the correct format (e.g., seconds)
        gt_end_time = gt_word['time']
        asr_gen_time = asr_word['time']
        
        latency = max(0.0, asr_gen_time - gt_end_time)
        total_latency += latency
        count += 1
    
    if count == 0:
        return 0.0  # Avoid division by zero
    
    average_latency = total_latency / count
    return average_latency


def load_gold(fname):
    '''load gold word-level timestamped transcript from the filename `fname`.

    In the file, there are tab-separated word-level timestamps in seconds: 
    Each word has its beginning and end timestamp in seconds, and then the word. Example:

    0.753	1.113	Hello,
    1.2429999999999999	1.443	this
    1.443	1.593	is
    1.593	1.833	Jiawei
    1.833	2.193	Zhou
    2.193	2.443	from

    It considers the end timestamps as the true timestamp of the words. 
    (Another options are the center, or the beginning. We assume it's arbitrary.)

    Return:
        a list of [{'word': 'Hello', 'time': 1.113}, ...]
    '''

    gold = fname

    with open(gold, "r") as f:
        gt_words = []
        for l in f:
            beg,end,word = l.split("\t")
            w = {"word": word.strip(), "time": float(end)}
            gt_words.append(w)
    return gt_words

def to_chars(words):
    '''convert words to chars'''
    chars = []
    for w in words:  
        for c in w["word"]:
            chars.append({"word": c, "time": w["time"]})
        # adding space.
        # TODO: this approach doesn't work for languages that don't use space, such as Chinese or Japanese.
        chars.append({"word": " ", "time": w["time"]})
    return chars


def load_asr_words(instream=sys.stdin):
    '''Loads the words of the asr transcript from the iterable instream (e.g. an opened file, like sys.stdin).

    Example:

        2600.0000 764 2600  Hello, this is
        4440.0000 2600 4440  Jiawei Zhou from Harvard
        6280.0000 4440 6280  University. I am very glad to present our

    There are space-separated emission timestamps in miliseconds, then beg and end timestamps, then a space, then a space if the last word ends with a space, or another character, and then a sequence of words emitted at the same time.
    Only the first  

    Return: the same as load_gold(). The miliseconds are converted to seconds.
    '''


    asr_ts_words = []
    for l in instream:
        ts, beg, end, *_ = l.split(" ")
        text = l[len(ts)+len(beg)+len(end)+3:-1]

        skip = 0
        words = text.split()
        if text[0] != " ":  # an ASR word in on multiple lines, we consider the timestamp of the last part
            asr_ts_words[-1] = (ts, asr_ts_words[-1][1]+words[0])
            skip = 1
        for w in words[skip:]:
            asr_ts_words.append((ts,w))

    # we reformat ts from ms to seconds
    # we make it asr words
    asr_words = [{"word": word, "time": float(ts)/1000} for ts, word in asr_ts_words]
    return asr_words



def char_to_word_alignment(alignment):
    words_alignment = []

    wa, wb = "", ""
    la = {}
    lb = {}
    for a,b in alignment:
        if a is not None:
            wa += a["word"]
            la = a
        if b is not None:
            wb += b["word"]
            lb = b
            if a is None and b["word"] == " ":  # insertion, ASR word ended
                b_w = {"word": wb, "time": lb["time"]}
                if wa != "" and wa != " ":
                    a_w = {"word":wa, "time": la["time"]}
                    wa = ""
                    la = {}
                else:
                    a_w = None
                words_alignment.append((a_w, b_w))
                wb = ""
                lb = {}

        if a is not None and a["word"] == " ":
            if lb == {} or wb == " ":
                b_w = None
            else:
                b_w = {"word": wb, "time": lb["time"]}
            words_alignment.append(({"word": wa, "time": la["time"]}, b_w))
            wa, wb = "", ""
            la = {}
            lb = {}
    return words_alignment

def main(debug=False):

    # load gold words
    gt_words = load_gold(sys.argv[1])
    # and convert them to characters
    gt_chars = to_chars(gt_words)

    # load asr transcripts
    asr_words = load_asr_words(sys.stdin)
    # and convert them to chars
    asr_chars = to_chars(asr_words)

    # Align
    alignment = continuous_levenshtein_alignment(gt_chars, asr_chars)

    if debug: # debug print: char alignments
        eprint("# char alignments:")
        eprint(r"# edit-operations \t gold \t asr \t latency diff or -1 if not available")
        eprint("# (note: sometimes there is a space that is is printed but not visible)")

        for g,a in alignment:
            if g is not None and a is not None:
                diff = a['time']-g['time']
                diff = "%.2f" % diff
                operation = "COPY" if g['word']==a['word'] else "SUB"
                eprint(operation,g['word'],a['word'],diff,sep="\t")
            elif g is not None:
                eprint("DEL",g['word'],"",-1,sep="\t")
            else:
                eprint("INS","",a['word'],-1,sep="\t")
        eprint("# char alignments ended.")
    words_alignment = char_to_word_alignment(alignment)

    if debug: # debug print -- word alignment
        eprint("# word alignments")
        eprint("# edit-operations, gold, asr, latency-diff or -1 if not available")
        for g,a in words_alignment:
            if g is not None and a is not None:
                diff = a['time']-g['time']
                diff = "%.2f" % diff
                operation = "COPY" if g['word']==a['word'] else "SUB"
                eprint(operation,g['word'],a['word'],diff,sep="\t")
            elif g is not None:
                eprint("DEL",g['word'],"",-1,sep="\t")
            else:
                eprint("INS","",a['word'],-1,sep="\t")
        eprint("# word alignments ended.")

    # Calculate average latency
    if debug:
        eprint("# length of word_alignment:",len(words_alignment))
        eprint("# length of gold words:",len(gt_words))
        eprint("# length of asr words:",len(asr_words))
    alignment = [p for p in words_alignment if None not in p]  # "None not in a" means removing deletions and insertions
    average_latency = calculate_latency(alignment)

    eprint(f"Average Latency: {average_latency} seconds")
    print(f"{average_latency}")
    return average_latency

if __name__ == "__main__":
    main(debug=len(sys.argv)>2)