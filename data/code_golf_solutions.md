## 1. 12-days-of-christmas
```python
def sing_12_days_of_christmas():
    """Prints the lyrics to the song The 12 Days of Christmas."""

    days = [
        "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
        "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth"
    ]

    gifts = [
        "A Partridge in a Pear Tree.",
        "Two Turtle Doves, and",
        "Three French Hens,",
        "Four Calling Birds,",
        "Five Gold Rings,",
        "Six Geese-a-Laying,",
        "Seven Swans-a-Swimming,",
        "Eight Maids-a-Milking,",
        "Nine Ladies Dancing,",
        "Ten Lords-a-Leaping,",
        "Eleven Pipers Piping,",
        "Twelve Drummers Drumming,"
    ]

    for i in range(12):
        print(f"On the {days[i]} day of Christmas")
        print("My true love sent to me")
        for j in range(i, -1, -1):
            print(gifts[j])
        print()

if __name__ == "__main__":
    sing_12_days_of_christmas()
```

---

## 2. 24-game
```python
from itertools import combinations_with_replacement, combinations
from fractions import Fraction

def solve_24_game():
    solvable = set()
    ops = [lambda a, b: a + b, 
           lambda a, b: a - b, 
           lambda a, b: b - a,
           lambda a, b: a * b,
           lambda a, b: a / b if b != 0 else None,
           lambda a, b: b / a if a != 0 else None]
    
    for quad in combinations_with_replacement(range(1, 14), 4):
        nums = [Fraction(n) for n in quad]
        if can_make_24(nums, ops):
            solvable.add(quad)
    
    for quad in sorted(solvable):
        print(' '.join(map(str, quad)))

def can_make_24(nums, ops, remaining=None):
    if remaining is None:
        remaining = set(range(4))
    
    if len(remaining) == 1:
        idx = next(iter(remaining))
        return abs(float(nums[idx]) - 24) < 1e-6
    
    for i, j in combinations(remaining, 2):
        new_remaining = remaining - {i, j}
        for op in ops:
            try:
                res = op(nums[i], nums[j])
                if res is None:  # division by zero
                    continue
                new_nums = nums.copy()
                new_nums[i] = res
                if can_make_24(new_nums, ops, new_remaining | {i}):
                    return True
            except ZeroDivisionError:
                continue
    return False

if __name__ == "__main__":
    solve_24_game()
```

---

## 3. 99-bottles-of-beer
```python
def ninety_nine_bottles():
    for i in range(99, 0, -1):
        bottles = lambda n: f"{n} bottle{'s'[:n!=1]} of beer"
        print(f"{bottles(i)} on the wall, {bottles(i)}.")
        print(f"Take one down and pass it around, {bottles(i-1) if i > 1 else 'no more bottles of beer'} on the wall.\n")
    print("No more bottles of beer on the wall, no more bottles of beer.")
    print("Go to the store and buy some more, 99 bottles of beer on the wall.")

ninety_nine_bottles()
```

---

## 4. abundant-numbers
```python
def is_abundant(n):
    """
    检查一个数是否为盈数。

    Args:
        n: 要检查的整数。

    Returns:
        如果 n 是盈数，则返回 True，否则返回 False。
    """
    if n <= 1:
        return False

    sum_of_divisors = 1
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            sum_of_divisors += i
            if i * i != n:
                sum_of_divisors += n // i

    return sum_of_divisors > n


def find_abundant_numbers(limit):
    """
    找到从 1 到 limit（包括 limit）的所有盈数。

    Args:
        limit: 搜索范围的上限。

    Returns:
        一个包含盈数的列表。
    """
    abundant_numbers = []
    for i in range(1, limit + 1):
        if is_abundant(i):
            abundant_numbers.append(i)
    return abundant_numbers


if __name__ == "__main__":
    abundant_numbers = find_abundant_numbers(200)
    for number in abundant_numbers:
        print(number)
```

---

## 5. abundant-numbers-long
```python
def is_abundant(n):
    """
    Checks if a number is abundant.

    Args:
        n: The number to check.

    Returns:
        True if the number is abundant, False otherwise.
    """
    if n <= 1:
        return False

    divisor_sum = 1
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisor_sum += i
            if i * i != n:
                divisor_sum += n // i

    return divisor_sum > n


def find_abundant_numbers(limit):
    """
    Finds all abundant numbers up to a given limit.

    Args:
        limit: The upper limit to search for abundant numbers.

    Returns:
        A list of abundant numbers.
    """
    abundant_numbers = []
    for i in range(1, limit + 1):
        if is_abundant(i):
            abundant_numbers.append(i)
    return abundant_numbers


if __name__ == "__main__":
    abundant_numbers = find_abundant_numbers(1000)
    for number in abundant_numbers:
        print(number)
```

---

## 6. arabic-to-roman
```python
def to_roman(num):
    roman_map = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
    ]
    result = []
    for value, symbol in roman_map:
        while num >= value:
            result.append(symbol)
            num -= value
    return ''.join(result)

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    for arg in sys.argv[1:]:
        try:
            num = int(arg)
            if 1 <= num <= 3999:
                print(to_roman(num))
            else:
                print(f"Error: {num} is out of range (1-3999)")
        except ValueError:
            print(f"Error: '{arg}' is not a valid integer")
```

---

## 7. arithmetic-numbers
```python
def is_arithmetic_number(n):
  """
  检查一个数是否是算术数。

  Args:
    n: 要检查的整数。

  Returns:
    如果 n 是算术数，则返回 True，否则返回 False。
  """
  divisors = []
  for i in range(1, n + 1):
    if n % i == 0:
      divisors.append(i)

  if not divisors: # n = 0
    return False

  average = sum(divisors) / len(divisors)
  return average == int(average)


def find_arithmetic_numbers(limit):
  """
  找出直到 limit 的所有算术数。

  Args:
    limit: 要搜索的上限（包括）。

  Returns:
    一个包含所有直到 limit 的算术数的列表。
  """
  arithmetic_numbers = []
  for i in range(1, limit + 1):
    if is_arithmetic_number(i):
      arithmetic_numbers.append(i)
  return arithmetic_numbers


if __name__ == "__main__":
  arithmetic_numbers = find_arithmetic_numbers(10000)
  for number in arithmetic_numbers:
    print(number)
```

---

## 8. arrows
```python
def main():
    import sys
    arrow_map = {'↙': (-1, -1), '↲': (-1, -1), '⇙': (-1, -1), '←': (-1, 0), '⇐': (-1, 0), '⇦': (-1, 0), '↖': (-1, 1), '↰': (-1, 1), '⇖': (-1, 1), '↓': (0, -1), '⇓': (0, -1), '⇩': (0, -1), '↔': (0, 0), '↕': (0, 0), '⇔': (0, 0), '⇕': (0, 0), '⥀': (0, 0), '⥁': (0, 0), '↑': (0, 1), '⇑': (0, 1), '⇧': (0, 1), '↘': (1, -1), '↳': (1, -1), '⇘': (1, -1), '→': (1, 0), '⇒': (1, 0), '⇨': (1, 0), '↗': (1, 1), '↱': (1, 1), '⇗': (1, 1)}
    x, y = 0, 0
    if len(sys.argv) > 1:
        input_str = ''.join(sys.argv[1:])
        for char in input_str:
            if char in arrow_map:
                dx, dy = arrow_map[char]
                x += dx
                y += dy
                print(f"{x} {y}")
    else:
        print("请提供Unicode箭头作为参数")

if __name__ == "__main__":
    main()
```

---

## 9. ascending-primes
```python
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0: return False
    return True

def find_ascending_primes():
    from itertools import combinations
    primes = []
    for length in range(1, 9):
        for digits in combinations('123456789', length):
            num = int(''.join(digits))
            if is_prime(num): primes.append(num)
    return primes

ascending_primes = find_ascending_primes()
for prime in ascending_primes: print(prime)
```

---

## 10. ascii-table
```python
def print_ascii_table():
    """Prints a hex ASCII table as described in the prompt."""

    print("   2 3 4 5 6 7")
    print(" -------------")
    for i in range(16):
        hex_val = hex(i)[2:].upper()  # Convert to hex and uppercase

        char_0 = chr(i + 0x20)  # Column 2 (0x20 to 0x2F)
        char_1 = chr(i + 0x30)  # Column 3 (0x30 to 0x3F)
        char_2 = chr(i + 0x40)  # Column 4 (0x40 to 0x4F)
        char_3 = chr(i + 0x50)  # Column 5 (0x50 to 0x5F)
        char_4 = chr(i + 0x60)  # Column 6 (0x60 to 0x6F)
        char_5 = chr(i + 0x70)  # Column 7 (0x70 to 0x7F)

        if i == 15:
            char_5 = "DEL"
        print(f"{hex_val}: {char_0} {char_1} {char_2} {char_3} {char_4} {char_5}")

if __name__ == "__main__":
    print_ascii_table()
```

---

## 11. billiards
```python
import sys
def draw_billiard_path(h,w):
 table=[[' 'for _ in range(w)]for _ in range(h)]
 x,y,dx,dy=0,0,1,1
 while True:
  table[y][x]='\\'if dx==dy else '/'
  x+=dx;y+=dy
  if x==w:x=w-1;dx*=-1
  elif x<0:x=0;dx*=-1
  if y==h:y=h-1;dy*=-1
  elif y<0:y=0;dy*=-1
  if x==0 and y==0 and dx==1 and dy==1:break
 for row in table:print(''.join(row))
 print()
raw_args=sys.argv[1:]
flattened=[]
for arg in raw_args:flattened.extend(arg.strip().split())
if len(flattened)%2!=0:sys.exit(1)
for i in range(0,len(flattened),2):
 h,w=int(flattened[i]),int(flattened[i+1])
 draw_billiard_path(h,w)
```

---

## 12. brainfuck
```python
def brainfuck_interpreter(program):
    """
    Interprets a Brainfuck program.

    Args:
        program: The Brainfuck program string.

    Returns:
        The output string of the program.
    """

    array_size = 30000  # Simulate an infinitely large array with a large size
    array = [0] * array_size
    ptr = 0
    output = ""
    loop_stack = []
    program_counter = 0

    while program_counter < len(program):
        instruction = program[program_counter]

        if instruction == '>':
            ptr = (ptr + 1) % array_size  # Wrap around
        elif instruction == '<':
            ptr = (ptr - 1) % array_size  # Wrap around
        elif instruction == '+':
            array[ptr] = (array[ptr] + 1) % 256  # Wrap around at 256
        elif instruction == '-':
            array[ptr] = (array[ptr] - 1) % 256  # Wrap around at 256
        elif instruction == '.':
            output += chr(array[ptr])
        elif instruction == '[':
            if array[ptr] == 0:
                # Find the matching ']'
                loop_count = 1
                temp_counter = program_counter + 1
                while temp_counter < len(program):
                    if program[temp_counter] == '[':
                        loop_count += 1
                    elif program[temp_counter] == ']':
                        loop_count -= 1
                        if loop_count == 0:
                            program_counter = temp_counter
                            break
                    temp_counter += 1
                else:
                    raise ValueError("Unmatched '['")  # Handle unmatched '['
            else:
                loop_stack.append(program_counter)
        elif instruction == ']':
            if array[ptr] != 0:
                program_counter = loop_stack[-1]
            else:
                loop_stack.pop()
        
        program_counter += 1

    return output

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            program = sys.argv[i]
            try:
                output = brainfuck_interpreter(program)
                print(output, end="")  # Print the output without extra newline
            except ValueError as e:
                print(f"Error: {e}")
    else:
        print("No Brainfuck program provided as argument.")
```

---

## 13. card-number-validation
```python
import sys
for card in sys.argv[1:]:
 d=card.replace(' ','')
 if len(d)==16 and d.isdigit():
  s=0
  for i,c in enumerate(d):
   n=int(c)
   if i%2==0:
    n*=2
    if n>9:n-=9
   s+=n
  if s%10==0:print(card)
```

---

## 14. catalan-numbers
```python
import math

def catalan(n):
    """
    Calculates the nth Catalan number.
    """
    if n < 0:
        return 0
    numerator = math.factorial(2 * n)
    denominator = math.factorial(n) * math.factorial(n) * (n + 1)
    return numerator // denominator

def main():
    """
    Prints the first 100 Catalan numbers.
    """
    for n in range(100):
        print(catalan(n))

if __name__ == "__main__":
    main()
```

---

## 15. catalans-constant
```python
import decimal
from decimal import Decimal,getcontext
getcontext().prec=1010
def arctan(x):
 total=Decimal(0)
 term=x
 n=0
 threshold=Decimal(10)**(-1010)
 while abs(term)>threshold:
  total+=term
  n+=1
  term=term*(-x*x)*(2*n-1)/(2*n+1)
 return total
x1=Decimal(1)/5
x2=Decimal(1)/239
pi_val=16*arctan(x1)-4*arctan(x2)
sqrt3=Decimal(3).sqrt()
x_val=2+sqrt3
ln_val=x_val.ln()
S=Decimal(0)
a=Decimal(1)
S+=a
for k in range(1,1701):
 k_dec=Decimal(k)
 numerator=k_dec*(2*k_dec-1)
 denominator=2*(2*k_dec+1)**2
 a=a*numerator/denominator
 S+=a
term1=(pi_val/8)*ln_val
term2=(Decimal(3)/8)*S
G_val=term1+term2
s=format(G_val,'.1000f')
print(s)
```

---

## 16. christmas-trees
```python
def print_tree(size):
 width=2*size-1
 for i in range(size):
  stars='*'*(2*i+1)
  print(' '+stars.center(width))
 print(' '+'*'.center(width))
for s in range(3,10):
 print_tree(s)
 print()
```

---

## 17. collatz
```python
def collatz_stopping_time(n):
  """
  Calculates the stopping time of a number n according to the Collatz conjecture.

  Args:
    n: A positive integer.

  Returns:
    The number of steps it takes for n to reach 1.
  """
  if n <= 0:
    raise ValueError("Input must be a positive integer.")

  steps = 0
  while n != 1:
    if n % 2 == 0:
      n = n // 2  # Integer division
    else:
      n = 3 * n + 1
    steps += 1
  return steps


if __name__ == "__main__":
  for i in range(1, 1001):
    stopping_time = collatz_stopping_time(i)
    print(stopping_time)
```

---

## 18. css-colors
```python
import sys
color_map={"indianred":"#cd5c5c","lightcoral":"#f08080","salmon":"#fa8072","darksalmon":"#e9967a","lightsalmon":"#ffa07a","red":"#ff0000","crimson":"#dc143c","firebrick":"#b22222","darkred":"#8b0000","pink":"#ffc0cb","lightpink":"#ffb6c1","hotpink":"#ff69b4","deeppink":"#ff1493","mediumvioletred":"#c71585","palevioletred":"#db7093","coral":"#ff7f50","tomato":"#ff6347","orangered":"#ff4500","darkorange":"#ff8c00","orange":"#ffa500","gold":"#ffd700","yellow":"#ffff00","lightyellow":"#ffffe0","lemonchiffon":"#fffacd","lightgoldenrodyellow":"#fafad2","papayawhip":"#ffefd5","moccasin":"#ffe4b5","peachpuff":"#ffdab9","palegoldenrod":"#eee8aa","khaki":"#f0e68c","darkkhaki":"#bdb76b","lavender":"#e6e6fa","thistle":"#d8bfd8","plum":"#dda0dd","violet":"#ee82ee","orchid":"#da70d6","fuchsia":"#ff00ff","magenta":"#ff00ff","mediumorchid":"#ba55d3","mediumpurple":"#9370db","blueviolet":"#8a2be2","darkviolet":"#9400d3","darkorchid":"#9932cc","darkmagenta":"#8b008b","purple":"#800080","indigo":"#4b0082","darkslateblue":"#483d8b","slateblue":"#6a5acd","mediumslateblue":"#7b68ee","rebeccapurple":"#663399","greenyellow":"#adff2f","chartreuse":"#7fff00","lawngreen":"#7cfc00","lime":"#00ff00","limegreen":"#32cd32","palegreen":"#98fb98","lightgreen":"#90ee90","springgreen":"#00ff7f","mediumspringgreen":"#00fa9a","mediumseagreen":"#3cb371","seagreen":"#2e8b57","forestgreen":"#228b22","green":"#008000","darkgreen":"#006400","yellowgreen":"#9acd32","olivedrab":"#6b8e23","olive":"#808000","darkolivegreen":"#556b2f","mediumaquamarine":"#66cdaa","darkseagreen":"#8fbc8f","lightseagreen":"#20b2aa","darkcyan":"#008b8b","teal":"#008080","aqua":"#00ffff","cyan":"#00ffff","lightcyan":"#e0ffff","paleturquoise":"#afeeee","aquamarine":"#7fffd4","turquoise":"#40e0d0","mediumturquoise":"#48d1cc","darkturquoise":"#00ced1","cadetblue":"#5f9ea0","steelblue":"#4682b4","lightsteelblue":"#b0c4de","powderblue":"#b0e0e6","lightblue":"#add8e6","skyblue":"#87ceeb","lightskyblue":"#87cefa","deepskyblue":"#00bfff","dodgerblue":"#1e90ff","cornflowerblue":"#6495ed","royalblue":"#4169e1","blue":"#0000ff","mediumblue":"#0000cd","darkblue":"#00008b","navy":"#000080","midnightblue":"#191970","cornsilk":"#fff8dc","blanchedalmond":"#ffebcd","bisque":"#ffe4c4","navajowhite":"#ffdead","wheat":"#f5deb3","burlywood":"#deb887","tan":"#d2b48c","rosybrown":"#bc8f8f","sandybrown":"#f4a460","goldenrod":"#daa520","darkgoldenrod":"#b8860b","peru":"#cd853f","chocolate":"#d2691e","saddlebrown":"#8b4513","sienna":"#a0522d","brown":"#a52a2a","maroon":"#800000","white":"#ffffff","snow":"#fffafa","honeydew":"#f0fff0","mintcream":"#f5fffa","azure":"#f0ffff","aliceblue":"#f0f8ff","ghostwhite":"#f8f8ff","whitesmoke":"#f5f5f5","seashell":"#fff5ee","beige":"#f5f5dc","oldlace":"#fdf5e6","floralwhite":"#fffaf0","ivory":"#fffff0","antiquewhite":"#faebd7","linen":"#faf0e6","lavenderblush":"#fff0f5","mistyrose":"#ffe4e1","gainsboro":"#dcdcdc","lightgray":"#d3d3d3","lightgrey":"#d3d3d3","silver":"#c0c0c0","darkgray":"#a9a9a9","darkgrey":"#a9a9a9","gray":"#808080","grey":"#808080","dimgray":"#696969","dimgrey":"#696969","lightslategray":"#778899","lightslategrey":"#778899","slategray":"#708090","slategrey":"#708090","darkslategray":"#2f4f4f","darkslategrey":"#2f4f4f","black":"#000000"}
args=sys.argv[1:]
for name in args:
 key=name.lower()
 print(color_map.get(key,"Unknown"))
```

---

## 19. cubes
```python
def draw_cube(n):
 V='█';D='╱';H='─';W='│'
 offset=n+1
 Wd=n*4+2
 lines=[]
 lines.append(' '*offset+V+H*(Wd-2)+V)
 for i in range(1,offset):
  indent=offset-i
  lines.append(' '*indent+D+' '*(Wd-2)+D+' '*(i-1)+W)
 lines.append(V+H*(Wd-2)+V+' '*(offset-1)+W)
 for j in range(n):
  if j<n-1:lines.append(W+' '*(Wd-2)+W+' '*(offset-1)+W)
  else:lines.append(W+' '*(Wd-2)+W+' '*(offset-1)+V)
 for i in range(1,n+1):
  lines.append(W+' '*(Wd-2)+W+' '*(offset-1-i)+D)
 lines.append(V+H*(Wd-2)+V)
 return '\n'.join(lines)
for size in range(1,8):
 print(draw_cube(size))
 print()
```

---

## 20. day-of-week
```python
import sys;import datetime
def parse_and_print(date_str):
 try:
  date_obj=datetime.datetime.strptime(date_str,'%Y-%m-%d')
  print(date_obj.strftime('%A'))
 except ValueError:
  print(f"{date_str}: Invalid date format (expected YYYY-MM-DD)")
for date_input in sys.argv[1:]:
 parse_and_print(date_input)
```

---

## 21. dfa-simulator
```python
import sys
def main():
 if len(sys.argv)<2:
  print(f"Usage: python {sys.argv[0]} <DFA blocks as args>",file=sys.stderr)
  sys.exit(1)
 raw_lines=[]
 for arg in sys.argv[1:]:
  raw_lines.extend(arg.split('\n'))
 lines=[ln.rstrip() for ln in raw_lines if ln.strip()]
 idx=0
 while idx<len(lines):
  alphabet=lines[idx].split()
  idx+=1
  transitions={}
  accept_states=set()
  initial_state=None
  while idx<len(lines) and not lines[idx].startswith('"'):
   parts=lines[idx].split()
   first=parts[0]
   if any(c.isdigit() for c in first):
    tag=first
    state=first.lstrip('>F')
    rest=parts[1:]
   else:
    tag=first
    state=parts[1]
    rest=parts[2:]
   if '>' in tag:initial_state=state
   if 'F' in tag:accept_states.add(state)
   if len(rest)!=len(alphabet):
    print(f"Error: state {state} has {len(rest)} transitions but alphabet size is {len(alphabet)}",file=sys.stderr)
    sys.exit(1)
   transitions[state]=dict(zip(alphabet,rest))
   idx+=1
  if initial_state is None:
   print("Error: no initial state specified",file=sys.stderr)
   sys.exit(1)
  if idx>=len(lines) or not lines[idx].startswith('"'):
   print("Error: missing input string line",file=sys.stderr)
   sys.exit(1)
  input_line=lines[idx]
  input_str=input_line[1:-1] if input_line.endswith('"') and input_line.startswith('"') else ''
  idx+=1
  current=initial_state
  for ch in input_str:
   current=transitions.get(current,{}).get(ch)
   if current is None:break
  result="Accept" if current in accept_states else "Reject"
  print(f"{current if current is not None else 'None'} {result}")
if __name__=='__main__':main()
```

---

## 22. diamonds
```python
MAX=9
for n in range(1,MAX+1):
 for i in range(1,n+1):
  left=''.join(str(j)for j in range(1,i))
  right=left[::-1]
  s=f"{left}{i}{right}"
  print(' '*(10-i)+s)
 for i in range(n-1,0,-1):
  left=''.join(str(j)for j in range(1,i))
  right=left[::-1]
  s=f"{left}{i}{right}"
  print(' '*(10-i)+s)
 if n!=MAX:print()
```

---

## 23. divisors
```python
def find_divisors_and_print(n):
  """
  Finds the positive divisors of a number and prints them on a single line.

  Args:
    n: The number to find divisors for.
  """
  divisors = []
  for i in range(1, n + 1):
    if n % i == 0:
      divisors.append(str(i))  # Convert to string for easy joining

  print(" ".join(divisors))


# Iterate from 1 to 100 and find divisors for each number
for num in range(1, 101):
  find_divisors_and_print(num)
```

---

## 24. emirp-numbers
```python
def sieve(n):
 is_prime=[False,False]+[True]*(n-1)
 for i in range(2,int(n**0.5)+1):
  if is_prime[i]:
   for j in range(i*i,n+1,i):is_prime[j]=False
 return is_prime
def reverse_int(x):
 return int(str(x)[::-1])
def main():
 limit=1000
 is_prime=sieve(limit)
 primes={i for i,v in enumerate(is_prime) if v}
 for p in sorted(primes):
  q=reverse_int(p)
  if q!=p and q in primes:print(p)
if __name__=='__main__':main()
```

---

## 25. emirp-numbers-long
```python
def is_evil(n):
 return bin(n).count('1')%2==0
def main():
 for n in range(0,1001):
  if is_evil(n):print(n)
if __name__=='__main__':main()
```

---

## 26. emojify
```python
import sys
m={":-D":"😀",":-)":"🙂",":-|":"😐",":-(":"🙁",":-\\":"😕",":-*":"😗",":-O":"😮",":-#":"🤐","':-D":"😅","':-(":"😓",":'-)":"😂",":'-(":"😢",":-P":"😛",";-P":"😜","X-P":"😝","X-)":"😆","O:-)":"😇",";-)":"😉",":-$":"😳",":-":"😶","B-)":"😎",":-J":"😏","}:-)":"😈","}:-(":"👿",":-@":"😡"}
for arg in sys.argv[1:]:
 print(m.get(arg,arg))
```

---

## 27. evil-numbers
```python
def is_evil(n):
    """
    Checks if a number is an evil number.

    An evil number is a non-negative number that has an even number of 1s in
    its binary expansion.
    """
    binary_representation = bin(n)[2:]  # Convert to binary and remove "0b" prefix
    count_of_ones = binary_representation.count('1')
    return count_of_ones % 2 == 0

def find_evil_numbers(limit):
    """
    Prints all evil numbers from 0 to limit inclusive.
    """
    for i in range(limit + 1):
        if is_evil(i):
            print(i)

if __name__ == "__main__":
    find_evil_numbers(50)
```

---

## 28. evil-numbers-long
```python
def is_evil(n):
 return bin(n).count('1')%2==0
def main():
 for n in range(0,1001):
  if is_evil(n):print(n)
if __name__=='__main__':main()
```

---

## 29. factorial-factorisation
```python
import math
def get_primes(n):
 sieve=[True]*(n+1)
 sieve[0:2]=[False,False]
 for i in range(2,int(n**0.5)+1):
  if sieve[i]:
   for j in range(i*i,n+1,i):sieve[j]=False
 return [i for i,is_prime in enumerate(sieve) if is_prime]
def prime_exponents_in_factorial(n):
 primes=get_primes(n)
 factors=[]
 for p in primes:
  exp=0
  k=p
  while k<=n:
   exp+=n//k
   k*=p
  factors.append((p,exp))
 return factors
def print_factorization(factors):
 parts=[]
 for p,e in factors:
  if e==1:parts.append(f"{p}")
  else:parts.append(f"{p}^{e}")
 print("*".join(parts))
factors=prime_exponents_in_factorial(1000)
print_factorization(factors)
```

---

## 30. farey-sequence
```python
def farey(n):
    """
    Generates the Farey sequence of order n.

    Args:
        n: The order of the Farey sequence.

    Returns:
        A list of tuples, where each tuple represents a fraction (numerator, denominator).
    """
    result = [(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        result.append((a, b))
    return result

if __name__ == '__main__':
    farey_sequence = farey(50)
    for num, den in farey_sequence:
        print(f"{num}/{den}")
```

---

## 31. fibonacci
```python
def fibonacci(n):
  """
  生成斐波那契数列的前 n 个数字。

  Args:
    n: 要生成的斐波那契数字的数量。

  Returns:
    一个包含斐波那契数列前 n 个数字的列表。
  """
  fib_list = []
  a = 0
  b = 1
  for _ in range(n):
    fib_list.append(a)
    a, b = b, a + b
  return fib_list

if __name__ == "__main__":
  fib_numbers = fibonacci(31)
  for num in fib_numbers:
    print(num)
```

---

## 32. fizz-buzz
```python
def fizzbuzz():
  """Prints numbers from 1 to 100, replacing multiples of 3 with "Fizz",
  multiples of 5 with "Buzz", and multiples of both with "FizzBuzz".
  """
  for i in range(1, 101):
    output = ""
    if i % 3 == 0:
      output += "Fizz"
    if i % 5 == 0:
      output += "Buzz"

    if output == "":
      print(i)
    else:
      print(output)

if __name__ == "__main__":
  fizzbuzz()
```

---

## 33. foo-fizz-buzz-bar
```python
def fizzbuzz():
  """
  Prints numbers from 1 to 1000, replacing multiples of 2, 3, 5, and 7 with
  "Foo", "Fizz", "Buzz", and "Bar" respectively. If a number is a multiple
  of multiple of these numbers, it prints all the corresponding replacements.
  """
  for i in range(1, 1001):
    output = ""
    if i % 2 == 0:
      output += "Foo"
    if i % 3 == 0:
      output += "Fizz"
    if i % 5 == 0:
      output += "Buzz"
    if i % 7 == 0:
      output += "Bar"

    if output == "":
      print(i)
    else:
      print(output)

if __name__ == "__main__":
  fizzbuzz()
```

---

## 34. forsyth-edwards-notation
```python
import sys
pieces={'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙','k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'}
for arg in sys.argv[1:]:
 fen=arg.split()[0]
 for row in fen.split('/'):
  line=''
  for c in row:
   if c.isdigit():line+=' '*int(c)
   else:line+=pieces[c]
  print(line)
 print()
```

---

## 35. fractions
```python
import sys,math
for arg in sys.argv[1:]:
 a,b=map(int,arg.split('/'))
 g=math.gcd(a,b)
 print(f'{a//g}/{b//g}')
```

---

## 36. game-of-life
```python
import sys
def next_gen(grid):
 out=[]
 for i in range(32):
  row=''
  for j in range(32):
   cnt=0
   for di in (-1,0,1):
    for dj in (-1,0,1):
     if di==0 and dj==0:continue
     ni,nj=i+di,j+dj
     if 0<=ni<32 and 0<=nj<32 and grid[ni][nj]=='#':cnt+=1
   row+='#' if (grid[i][j]=='#' and cnt in (2,3))or(grid[i][j]=='.' and cnt==3)else '.'
  out.append(row)
 return out
for arg in sys.argv[1:]:
 lines=arg.splitlines()
 if len(lines)==1 and len(lines[0])==1024:grid=[list(lines[0][i*32:(i+1)*32])for i in range(32)]
 else:grid=[list(lines[i])for i in range(32)]
 for ln in next_gen(grid):print(ln)
```

---

## 37. gijswijts-sequence
```python
def gijswijt(n_terms):
 seq=[1]
 for m in range(1,n_terms):
  max_k=1
  for L in range(1,m+1):
   block=seq[-L:]
   k=0
   while L*(k+1)<=m and seq[-L*(k+1):]==block*(k+1):k+=1
   if k>max_k:max_k=k
  seq.append(max_k)
 return seq
if __name__=='__main__':
 for x in gijswijt(1000):print(x)
```

---

## 38. happy-numbers
```python
def is_happy(n):
    """
    判断一个数字是否是快乐数。

    Args:
        n: 要检查的数字。

    Returns:
        如果数字是快乐数，则返回 True，否则返回 False。
    """
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        sum_of_squares = 0
        while n > 0:
            digit = n % 10
            sum_of_squares += digit ** 2
            n //= 10
        n = sum_of_squares
    return n == 1


def find_happy_numbers(limit):
    """
    查找并打印从 1 到 limit（包括 limit）的所有快乐数。

    Args:
        limit: 要搜索到的上限。
    """
    for i in range(1, limit + 1):
        if is_happy(i):
            print(i)


if __name__ == "__main__":
    find_happy_numbers(200)
```

---

## 39. happy-numbers-long
```python
def sum_of_squares(n):
  """计算一个数的各位数字的平方和。"""
  sum_sq = 0
  while n > 0:
    digit = n % 10
    sum_sq += digit * digit
    n //= 10
  return sum_sq

def is_happy(n):
  """判断一个数是否是快乐数。"""
  seen = set()
  while n != 1 and n not in seen:
    seen.add(n)
    n = sum_of_squares(n)
  return n == 1

# 打印 1 到 1000 之间的所有快乐数。
for i in range(1, 1001):
  if is_happy(i):
    print(i)
```

---

## 40. hexdump
```python
import sys
def hexdump(data):
 for offset in range(0,len(data),16):
  chunk=data[offset:offset+16]
  prefix=f"{offset:08x}: "
  groups=[]
  for i in range(8):
   pair=chunk[2*i:2*i+2]
   if len(pair)==2:groups.append(f"{pair[0]:02x}{pair[1]:02x}")
   elif len(pair)==1:groups.append(f"{pair[0]:02x}  ")
   else:groups.append("    ")
  hex_part=" ".join(groups)
  pad=51-(len(prefix)+len(hex_part))
  ascii_part="".join((chr(b)if 32<=b<127 and b!=10 else".")for b in chunk)
  print(prefix+hex_part+" "*pad+ascii_part)
if __name__=='__main__':
 for arg in sys.argv[1:]:
  data=arg.encode("utf-8")
  hexdump(data)
  print()
```

---

## 41. intersection
```python
import sys
tokens=[]
for arg in sys.argv[1:]:tokens.extend(arg.split())
nums=list(map(int,tokens))
for i in range(0,len(nums),8):
 x1,y1,w1,h1,x2,y2,w2,h2=nums[i:i+8]
 overlap_w=max(0,min(x1+w1,x2+w2)-max(x1,x2))
 overlap_h=max(0,min(y1+h1,y2+h2)-max(y1,y2))
 print(overlap_w*overlap_h)
```

---

## 42. inventory-sequence
```python
def inventory_sequence(n):
 seq=[0]
 while len(seq)<n:
  k=0
  while True:
   c=seq.count(k)
   seq.append(c)
   if c==0 or len(seq)>=n:break
   k+=1
 return seq[:n]
if __name__=='__main__':
 for x in inventory_sequence(1000):print(x)
```

---

## 43. isbn
```python
import sys
def calc_check(s):
 ds=[int(c)for c in s if c.isdigit()][:9]
 total=sum((10-i)*d for i,d in enumerate(ds))
 x=(-total)%11
 return 'X' if x==10 else str(x)
for arg in sys.argv[1:]:
 chk=calc_check(arg)
 print(arg+chk)
```

---

## 44. jacobi-symbol
```python
import sys
def jacobi(a,n):
 if n<=0 or n%2==0:return 0
 a%=n
 r=1
 while a:
  while a%2==0:
   a//=2
   if n%8 in(3,5):r=-r
  a,n=n,a
  if a%4==n%4==3:r=-r
  a%=n
 return r if n==1 else 0
for arg in sys.argv[1:]:
 a,n=map(int,arg.split())
 print(jacobi(a,n))
```

---

## 45. kaprekar-numbers(Time limit exceeded)
```python
def kaprekar_numbers(limit):
    """
    Prints all Kaprekar numbers from 1 to limit inclusive.

    Args:
        limit: The upper limit for finding Kaprekar numbers.
    """

    for num in range(1, limit + 1):
        if num == 1:
            print(num)
            continue

        square = num * num
        square_str = str(square)
        length = len(square_str)

        for i in range(1, length):
            right = int(square_str[i:])
            left = int(square_str[:i])

            if left == 0 or right == 0:
                continue

            if left + right == num:
                print(num)
                break  # Move to the next number

if __name__ == "__main__":
    kaprekar_numbers(25000000)
```

---

## 46. kolakoski-constant
```python
from decimal import Decimal,getcontext
getcontext().prec=1005
def g(n):
 s=[1,2,2]
 i=2
 while len(s)<n:
  s+=[1 if s[-1]==2 else 2]*s[i]
  i+=1
 return s[:n]
k=g(5000)
b=int(''.join(str(x-1)for x in k[1:]),2)
d=Decimal(b)/2**(len(k)-1)
print(str(d)[:1002])
```

---

## 47. kolakoski-sequence
```python
def generate_kolakoski_sequence(length):
 sequence=[1,2,2]
 index=2
 while len(sequence)<length:
  next_value=3-sequence[-1]
  repeat=sequence[index]
  sequence.extend([next_value]*repeat)
  index+=1
 return sequence[:length]
def main():
 length=1000
 kolakoski_seq=generate_kolakoski_sequence(length)
 print(' '.join(map(str,kolakoski_seq)))
if __name__=='__main__':main()
```

---

## 48. leap-years
```python
def is_leap_year(year):
    """
    Checks if a given year is a leap year according to the Gregorian calendar rules.
    """
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def print_leap_years(start_year, end_year):
    """
    Prints all leap years between start_year and end_year (inclusive).
    """
    for year in range(start_year, end_year + 1):
        if is_leap_year(year):
            print(year)

if __name__ == "__main__":
    print_leap_years(1800, 2400)
```

---

## 49. levenshtein-distance
```python
import sys
def levenshtein_distance(s1,s2):
 m,n=len(s1),len(s2)
 dp=[[0]*(n+1)for _ in range(m+1)]
 for i in range(m+1):dp[i][0]=i
 for j in range(n+1):dp[0][j]=j
 for i in range(1,m+1):
  for j in range(1,n+1):
   cost=0 if s1[i-1]==s2[j-1]else 1
   dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+cost)
 return dp[m][n]
def main():
 args=sys.argv[1:]
 for arg in args:
  words=arg.split()
  if len(words)!=2:continue
  word1,word2=words[0],words[1]
  distance=levenshtein_distance(word1,word2)
  print(distance)
if __name__=='__main__':main()
```

---

## 50. leyland-numbers
```python
def solve_leyland_numbers():
 leyland_numbers=set()
 max_limit=100_000_000_000
 for y in range(2,12):
  for x in range(y,320000):
   number=x**y+y**x
   if number>max_limit:break
   leyland_numbers.add(number)
 sorted_numbers=sorted(list(leyland_numbers))
 for number in sorted_numbers:print(number)
if __name__=='__main__':solve_leyland_numbers()
```

---

## 51. ln-2
```python
import decimal
from decimal import Decimal,getcontext
getcontext().prec=1010
ln2=Decimal(2).ln()
ln2_str=format(ln2,'.1010f')
print(ln2_str[0:1002])
```

---

## 52. look-and-say
```python
def look_and_say(s):
    """
    Generates the next term in the Look and Say sequence.

    Args:
        s: The previous term in the sequence (as a string).

    Returns:
        The next term in the sequence (as a string).
    """
    result = ""
    count = 1
    for i in range(len(s)):
        if i + 1 < len(s) and s[i] == s[i + 1]:
            count += 1
        else:
            result += str(count) + s[i]
            count = 1
    return result

# Generate and print the first 20 terms of the Look and Say sequence
start = "1"
for i in range(20):
    print(start)
    start = look_and_say(start)
```

---

## 53. lucky-numbers
```python
def find_lucky_numbers(n):
 lucky_numbers_sequence=list(range(1,40000,2))
 k=2
 while k<len(lucky_numbers_sequence):
  p=lucky_numbers_sequence[k-1]
  lucky_numbers_sequence=[num for index,num in enumerate(lucky_numbers_sequence)if(index+1)%p!=0]
  k+=1
 return lucky_numbers_sequence[:n]
first_1000_lucky_numbers=find_lucky_numbers(1000)
for number in first_1000_lucky_numbers:print(number)
```

---

## 54. lucky-tickets
```python
import sys
def count_lucky_tickets(d,b):
 half_digits=d//2
 max_sum=half_digits*(b-1)
 dp=[[0]*(max_sum+1)for _ in range(half_digits+1)]
 dp[0][0]=1
 for i in range(1,half_digits+1):
  for j in range(max_sum+1):
   for k in range(b):
    if j-k>=0:dp[i][j]+=dp[i-1][j-k]
 half_ticket_ways=dp[half_digits]
 total_lucky_tickets=sum(ways**2 for ways in half_ticket_ways)
 return total_lucky_tickets
def main():
 if len(sys.argv)<2:return
 for arg in sys.argv[1:]:
  try:
   d_str,b_str=arg.split()
   d=int(d_str)
   b=int(b_str)
   if not(2<=d<=14 and d%2==0):print(f"错误: 票号位数 d={d} 必须是2到14之间的偶数。");continue
   if not(2<=b<=16):print(f"错误: 进制 b={b} 必须是2到16之间。");continue
   result=count_lucky_tickets(d,b)
   print(result)
  except ValueError:print(f"错误: 参数 '{arg}' 格式不正确，应为 'd b'。")
if __name__=='__main__':main()
```

---

## 55. mahjong
```python
import sys
from collections import Counter
CHARACTERS="🀇🀈🀉🀊🀋🀌🀍🀎🀏"
BAMBOO="🀐🀑🀒🀓🀔🀕🀖🀗🀘"
CIRCLES="🀙🀚🀛🀜🀝🀞🀟🀠🀡"
HONORS="🀀🀁🀂🀃🀄🀅🀆"
SUITS=[CHARACTERS,BAMBOO,CIRCLES]
ALL_TILES=HONORS+''.join(SUITS)
def get_first_tile(counts):
 for tile in ALL_TILES:
  if counts[tile]>0:return tile
 return None
def is_number_tile(tile):
 return tile in CHARACTERS or tile in BAMBOO or tile in CIRCLES
def get_next_tile(tile,suit_str):
 try:
  index=suit_str.index(tile)
  if index<8:return suit_str[index+1]
 except ValueError:pass
 return None
def can_form_melds(counts):
 first_tile=get_first_tile(counts)
 if not first_tile:return True
 if counts[first_tile]>=3:
  new_counts=counts.copy()
  new_counts[first_tile]-=3
  if can_form_melds(new_counts):return True
 if is_number_tile(first_tile):
  for suit_str in SUITS:
   if first_tile in suit_str:
    next_tile=get_next_tile(first_tile,suit_str)
    if next_tile and counts[next_tile]>=1:
     next_next_tile=get_next_tile(next_tile,suit_str)
     if next_next_tile and counts[next_next_tile]>=1:
      new_counts=counts.copy()
      new_counts[first_tile]-=1
      new_counts[next_tile]-=1
      new_counts[next_next_tile]-=1
      if can_form_melds(new_counts):return True
    break
 return False
def is_standard_complete(counts):
 if sum(counts.values())!=14:return False
 for tile in counts:
  if counts[tile]>=2:
   new_counts=counts.copy()
   new_counts[tile]-=2
   if can_form_melds(new_counts):return True
 return False
def is_seven_pairs(counts):
 return len(counts)==7 and all(c==2 for c in counts.values())
def is_thirteen_orphans(counts):
 thirteen_orphans_tiles=HONORS+CHARACTERS[0]+CHARACTERS[8]+BAMBOO[0]+BAMBOO[8]+CIRCLES[0]+CIRCLES[8]
 if len(counts)!=13:return False
 if any(tile not in thirteen_orphans_tiles for tile in counts):return False
 pair_count=0
 for c in counts.values():
  if c==2:pair_count+=1
  elif c!=1:return False
 return pair_count==1
def is_complete(hand_str):
 counts=Counter(hand_str)
 if sum(counts.values())!=14:return False
 if is_seven_pairs(counts):return True
 if is_thirteen_orphans(counts):return True
 if is_standard_complete(counts):return True
 return False
def main():
 if len(sys.argv)<2:return
 for hand_str in sys.argv[1:]:
  if is_complete(hand_str):print(hand_str)
if __name__=='__main__':main()
```

---

## 56. mandelbrot
```python
def draw_mandelbrot_set():
 width=81
 height=41
 max_iterations=1063
 min_real=-2.0
 max_real=0.5
 min_imag=-1.0
 max_imag=1.0
 for i in range(height):
  for j in range(width):
   c_real=min_real+j*(max_real-min_real)/(width-1)
   c_imag=max_imag-i*(max_imag-min_imag)/(height-1)
   c=complex(c_real,c_imag)
   a=complex(0,0)
   is_in_set=True
   for _ in range(max_iterations):
    a=a**2+c
    if abs(a)>2:is_in_set=False;break
   print('█'if is_in_set else'▒',end='')
  print()
if __name__=='__main__':draw_mandelbrot_set()
```

---

## 57. maze
```python
import sys
from collections import deque
def find_shortest_path(maze_string):
 maze_grid=[list(row)for row in maze_string.strip().split('\n')]
 rows,cols=len(maze_grid),len(maze_grid[0])
 start_pos,end_pos=None,None
 for r in range(rows):
  for c in range(cols):
   if maze_grid[r][c]=='S':start_pos=(r,c)
   elif maze_grid[r][c]=='E':end_pos=(r,c)
 if not start_pos or not end_pos:return "错误: 迷宫中未找到起点或终点。"
 queue=deque([start_pos])
 visited={start_pos}
 parent={start_pos:None}
 path_found=False
 while queue:
  r,c=queue.popleft()
  if(r,c)==end_pos:path_found=True;break
  for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
   nr,nc=r+dr,c+dc
   if 0<=nr<rows and 0<=nc<cols and maze_grid[nr][nc]!='#'and(nr,nc)not in visited:
    visited.add((nr,nc))
    parent[(nr,nc)]=(r,c)
    queue.append((nr,nc))
 if path_found:
  path=[]
  current=end_pos
  while current:
   path.append(current)
   current=parent[current]
  result_grid=[list(row)for row in maze_string.strip().split('\n')]
  for r,c in path:
   if result_grid[r][c]not in('S','E'):result_grid[r][c]='.'
  return '\n'.join(''.join(row)for row in result_grid)
 else:return "未找到路径。"
def main():
 if len(sys.argv)<2:print("请提供一个或多个迷宫字符串作为命令行参数。");return
 for i,maze_string in enumerate(sys.argv[1:]):
  if i>0:print("-"*20)
  result=find_shortest_path(maze_string)
  print(result)
if __name__=='__main__':main()
```

---

## 58. medal-tally
```python
import sys
from collections import Counter
def assign_medals_final(scores_str):
 try:scores=[int(s)for s in scores_str.split()]
 except ValueError:return f"错误：输入 '{scores_str}' 包含无效字符。"
 if not scores:return ""
 unique_scores=sorted(list(set(scores)))
 all_awarded_medals=[]
 current_rank=1
 for score_value in unique_scores:
  if current_rank>3:break
  score_count=scores.count(score_value)
  if current_rank==1:
   if score_count==1:
    all_awarded_medals.append('💎')
    all_awarded_medals.append('🥇')
   else:all_awarded_medals.extend(['🥇']*score_count)
  elif current_rank==2:all_awarded_medals.extend(['🥈']*score_count)
  elif current_rank==3:all_awarded_medals.extend(['🥉']*score_count)
  current_rank+=score_count
 if not all_awarded_medals:return ""
 medal_counts=Counter(all_awarded_medals)
 output_parts=[]
 medal_order=['💎','🥇','🥈','🥉']
 for medal in medal_order:
  if medal_counts[medal]>0:output_parts.append(f"{medal_counts[medal]}{medal}")
 return ' '.join(output_parts)
def main():
 if len(sys.argv)<2:print("用法: python script.py \"<分数列表1>\" \"<分数列表2>\" ...");return
 for scores_argument in sys.argv[1:]:
  result=assign_medals_final(scores_argument)
  print(result)
if __name__=='__main__':main()
```

---

## 59. morse-decoder
```python
import sys
def morse_decode(morse_code_str):
 MORSE_CODE_DICT={'.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F','--.':'G','....':'H','..':'I','.---':'J','-.-':'K','.-..':'L','--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R','...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y','--..':'Z','-----':'0','.----':'1','..---':'2','...--':'3','....-':'4','.....':'5','-....':'6','--...':'7','---..':'8','----.':'9'}
 normalized_code=morse_code_str.replace('▄▄▄','-').replace('▄','.')
 words_morse=normalized_code.split('          ')
 decoded_words=[]
 for word_morse in words_morse:
  letters_morse=word_morse.split('   ')
  decoded_letters=[]
  for letter_morse in letters_morse:
   morse_unit=letter_morse.strip().replace(' ','');decoded_letters.append(MORSE_CODE_DICT.get(morse_unit,''))
  decoded_words.append(''.join(decoded_letters))
 return ' '.join(decoded_words)
if __name__=='__main__':
 if len(sys.argv)>1:
  for arg in sys.argv[1:]:print(morse_decode(arg))
```

---

## 60. morse-encoder
```python
import sys
def morse_encode(text):
 MORSE_CODE_DICT={'A':'▄ ▄▄▄','B':'▄▄▄ ▄ ▄ ▄','C':'▄▄▄ ▄ ▄▄▄ ▄','D':'▄▄▄ ▄ ▄','E':'▄','F':'▄ ▄ ▄▄▄ ▄','G':'▄▄▄ ▄▄▄ ▄','H':'▄ ▄ ▄ ▄','I':'▄ ▄','J':'▄ ▄▄▄ ▄▄▄ ▄▄▄','K':'▄▄▄ ▄ ▄▄▄','L':'▄ ▄▄▄ ▄ ▄','M':'▄▄▄ ▄▄▄','N':'▄▄▄ ▄','O':'▄▄▄ ▄▄▄ ▄▄▄','P':'▄ ▄▄▄ ▄▄▄ ▄','Q':'▄▄▄ ▄▄▄ ▄ ▄▄▄','R':'▄ ▄▄▄ ▄','S':'▄ ▄ ▄','T':'▄▄▄','U':'▄ ▄ ▄▄▄','V':'▄ ▄ ▄ ▄▄▄','W':'▄ ▄▄▄ ▄▄▄','X':'▄▄▄ ▄ ▄ ▄▄▄','Y':'▄▄▄ ▄ ▄▄▄ ▄▄▄','Z':'▄▄▄ ▄▄▄ ▄ ▄','0':'▄▄▄ ▄▄▄ ▄▄▄ ▄▄▄ ▄▄▄','1':'▄ ▄▄▄ ▄▄▄ ▄▄▄ ▄▄▄','2':'▄ ▄ ▄▄▄ ▄▄▄ ▄▄▄','3':'▄ ▄ ▄ ▄▄▄ ▄▄▄','4':'▄ ▄ ▄ ▄ ▄▄▄','5':'▄ ▄ ▄ ▄ ▄','6':'▄▄▄ ▄ ▄ ▄ ▄','7':'▄▄▄ ▄▄▄ ▄ ▄ ▄','8':'▄▄▄ ▄▄▄ ▄▄▄ ▄ ▄','9':'▄▄▄ ▄▄▄ ▄▄▄ ▄▄▄ ▄',' ':' '}
 encoded_words=[]
 for word in text.split(' '):
  encoded_letters=[]
  for char in word.upper():
   if char in MORSE_CODE_DICT:encoded_letters.append(MORSE_CODE_DICT[char])
  encoded_words.append('   '.join(encoded_letters))
 return '          '.join(encoded_words)
if __name__=='__main__':
 if len(sys.argv)>1:
  for arg in sys.argv[1:]:print(morse_encode(arg))
```

---

## 61. musical-chords
```python
import sys
from collections import defaultdict
note_to_pitch={'A':0,'A♯':1,'B♭':1,'B':2,'C♭':2,'B♯':3,'C':3,'C♯':4,'D♭':4,'D':5,'D♯':6,'E♭':6,'E':7,'F♭':7,'E♯':8,'F':8,'F♯':9,'G♭':9,'G':10,'G♯':11,'A♭':11}
letter_index={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
chord_types={(3,3):'°',(3,4):'m',(4,3):'',(4,4):'+'}
def interval(a,b):return(b-a)%12
def is_root(root_letter,others):
 li=sorted((letter_index[c[0]]-letter_index[root_letter])%7 for c in others)
 return li==[2,4]
def main():
 for triad in sys.argv[1:]:
  notes=triad.strip().split()
  pitch_map={n:note_to_pitch[n]for n in notes}
  result=None
  for root in notes:
   others=[n for n in notes if n!=root]
   if is_root(root[0],others):
    r,o1,o2=root,others[0],others[1]
    ordered=sorted([(letter_index[n[0]]-letter_index[r[0]])%7,n]for n in[o1,o2])
    first=ordered[0][1]
    second=ordered[1][1]
    i1=interval(pitch_map[r],pitch_map[first])
    i2=interval(pitch_map[first],pitch_map[second])
    chord=r+chord_types.get((i1,i2),'?')
    result=chord
    break
  print(result if result else'Unknown')
if __name__=='__main__':main()
```

---

## 62. n-queens
```python
def solve_n_queens(n):
 def backtrack(col,diagonals,anti_diagonals,rows,state):
  if col==n:solutions.append(''.join(str(r+1)for r in state));return
  for row in range(n):
   if row in rows or(row-col)in diagonals or(row+col)in anti_diagonals:continue
   rows.add(row);diagonals.add(row-col);anti_diagonals.add(row+col);state.append(row)
   backtrack(col+1,diagonals,anti_diagonals,rows,state)
   rows.remove(row);diagonals.remove(row-col);anti_diagonals.remove(row+col);state.pop()
 solutions=[]
 backtrack(0,set(),set(),set(),[])
 return solutions
for n in range(4,9):
 for sol in solve_n_queens(n):print(sol)
```

---

## 63. niven-numbers
```python
def is_niven(num):
  """
  Checks if a number is a Niven number.

  Args:
    num: The number to check.

  Returns:
    True if the number is a Niven number, False otherwise.
  """
  num_str = str(num)
  digit_sum = sum(int(digit) for digit in num_str)

  if digit_sum == 0:
      return False

  return num % digit_sum == 0


def print_niven_numbers(limit):
  """
  Prints all Niven numbers from 1 to the given limit (inclusive).

  Args:
    limit: The upper limit for finding Niven numbers.
  """
  for i in range(1, limit + 1):
    if is_niven(i):
      print(i)

if __name__ == "__main__":
  print_niven_numbers(100)
```

---

## 64. niven-numbers-long
```python
def is_niven(num):
    """
    Checks if a number is a Niven number.

    Args:
        num: The number to check.

    Returns:
        True if the number is a Niven number, False otherwise.
    """
    num_str = str(num)
    digit_sum = sum(int(digit) for digit in num_str)
    if digit_sum == 0:  # Handle the case where digit_sum is zero to avoid division by zero
        return False
    return num % digit_sum == 0


def find_niven_numbers(limit):
    """
    Prints all Niven numbers from 1 to the given limit, inclusive.

    Args:
        limit: The upper limit for finding Niven numbers.
    """
    for i in range(1, limit + 1):
        if is_niven(i):
            print(i)


if __name__ == "__main__":
    find_niven_numbers(10000)
```

---

## 65. number-spiral
```python
def spiral_grid(n):
    """
    Generates and prints a spiral grid of numbers from 0 to n*n - 1.

    Args:
        n: The size of the grid (n x n).
    """

    grid = [[0] * n for _ in range(n)]
    num = 0
    top, bottom = 0, n - 1
    left, right = 0, n - 1
    direction = 0  # 0: right, 1: down, 2: left, 3: up

    while top <= bottom and left <= right:
        if direction == 0:  # right
            for i in range(left, right + 1):
                grid[top][i] = num
                num += 1
            top += 1
        elif direction == 1:  # down
            for i in range(top, bottom + 1):
                grid[i][right] = num
                num += 1
            right -= 1
        elif direction == 2:  # left
            for i in range(right, left - 1, -1):
                grid[bottom][i] = num
                num += 1
            bottom -= 1
        elif direction == 3:  # up
            for i in range(bottom, top - 1, -1):
                grid[i][left] = num
                num += 1
            left += 1

        direction = (direction + 1) % 4

    for row in grid:
        for val in row:
            print(f"{val:>2}", end=" ")
        print()

if __name__ == "__main__":
    spiral_grid(10)
```

---

## 66. odious-numbers
```python
def is_odious(n):
  """
  Checks if a number is odious.

  An odious number is a non-negative number that has an odd number of 1s in
  its binary expansion.

  Args:
    n: The number to check.

  Returns:
    True if the number is odious, False otherwise.
  """
  binary_representation = bin(n)[2:]  # Convert to binary string and remove "0b" prefix
  count_of_ones = binary_representation.count('1')
  return count_of_ones % 2 != 0


if __name__ == "__main__":
  for i in range(51):
    if is_odious(i):
      print(i)
```

---

## 67. odious-numbers-long
```python
def is_odious(n):
  """
  Checks if a number is odious (has an odd number of 1s in its binary representation).

  Args:
    n: The non-negative integer to check.

  Returns:
    True if the number is odious, False otherwise.
  """
  binary_representation = bin(n)[2:]  # Convert to binary and remove "0b" prefix
  count_of_ones = binary_representation.count('1')
  return count_of_ones % 2 != 0


def print_odious_numbers(limit):
  """
  Prints all odious numbers from 0 to limit (inclusive).

  Args:
    limit: The upper limit (inclusive) for finding odious numbers.
  """
  for i in range(limit + 1):
    if is_odious(i):
      print(i)


if __name__ == "__main__":
  print_odious_numbers(1000)
```

---

## 68. ordinal-numbers
```python
import sys
def ordinal(n):
 if 11<=(n%100)<=13:suffix="th"
 else:suffix={1:"st",2:"nd",3:"rd"}.get(n%10,"th")
 return f"{n}{suffix}"
def main():
 for arg in sys.argv[1:]:
  try:
   n=int(arg)
   if 0<=n<=999:print(ordinal(n))
   else:print(f"Error: {n} is out of range (0-999)")
  except ValueError:print(f"Error: '{arg}' is not a valid integer")
if __name__=='__main__':main()
```

---

## 69. palindromemordnilap
```python
import sys
def make_palindrome(s):
 r=s[::-1]
 for k in range(len(s),-1,-1):
  cand=s+r[k:]
  if cand==cand[::-1]:return cand
 return s+r
def main():
 if len(sys.argv)<2:print("Usage: python script.py str1 [str2 ...]");return
 for s in sys.argv[1:]:print(make_palindrome(s))
if __name__=='__main__':main()
```

---

## 70. pangram-grep
```python
import sys

def is_pangram(sentence):
    """
    Checks if a sentence is a pangram (contains all letters from A to Z, case-insensitive).

    Args:
        sentence: The sentence to check.

    Returns:
        True if the sentence is a pangram, False otherwise.
    """
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    sentence = sentence.lower()
    letters_present = set()
    for char in sentence:
        if 'a' <= char <= 'z':
            letters_present.add(char)
    return letters_present == alphabet

if __name__ == "__main__":
    sentences = sys.argv[1:]  # Get sentences from command line arguments

    for sentence in sentences:
        if is_pangram(sentence):
            print(sentence)
```

---

## 71. partition-numbers
```python
def partition_numbers(n):
 p=[1]+[0]*n
 for k in range(1,n+1):
  for i in range(k,n+1):p[i]+=p[i-k]
 return p
p=partition_numbers(99)
for i,val in enumerate(p):print(val)
```

---

## 72. pascals-triangle
```python
def pascal_triangle(rows):
 triangle=[[1]]
 for i in range(1,rows):
  prev=triangle[-1]
  row=[1]+[prev[j]+prev[j+1]for j in range(len(prev)-1)]+[1]
  triangle.append(row)
 return triangle
for row in pascal_triangle(20):print(' '.join(map(str,row)))
```

---

## 73. pernicious-numbers
```python
def is_prime(n):
    """
    Check if a number is prime.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_set_bits(n):
    """
    Count the number of set bits (1s) in the binary representation of a number.
    """
    count = 0
    while (n > 0):
        n &= (n - 1)
        count += 1
    return count

if __name__ == "__main__":
    for i in range(0, 51):
        set_bits = count_set_bits(i)
        if is_prime(set_bits):
            print(i)
```

---

## 74. pernicious-numbers-long
```python
def is_prime(n):
    """
    Checks if a number is prime.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def count_set_bits(n):
    """
    Counts the number of set bits (1s) in the binary representation of a number.
    """
    count = 0
    while (n > 0):
        n &= (n - 1)
        count += 1
    return count

def find_pernicious_numbers(limit):
    """
    Prints all pernicious numbers from 0 to limit inclusive.
    """
    for i in range(limit + 1):
        set_bits = count_set_bits(i)
        if is_prime(set_bits):
            print(i)

if __name__ == "__main__":
    find_pernicious_numbers(10000)
```

---

## 75. poker
```python
import sys
from collections import Counter
UNICODE_MAP={'🂡':'A','🂢':'2','🂣':'3','🂤':'4','🂥':'5','🂦':'6','🂧':'7','🂨':'8','🂩':'9','🂪':'T','🂫':'J','🂭':'Q','🂮':'K','🂵':'5','🂱':'A','🂲':'2','🂳':'3','🂴':'4','🂵':'5','🂶':'6','🂷':'7','🂸':'8','🂹':'9','🂺':'T','🂻':'J','🂽':'Q','🂾':'K','🂿':'A','🃁':'A','🃂':'2','🃃':'3','🃄':'4','🃅':'5','🃆':'6','🃇':'7','🃈':'8','🃉':'9','🃊':'T','🃋':'J','🃍':'Q','🃎':'K','🃑':'A','🃒':'2','🃓':'3','🃔':'4','🃕':'5','🃖':'6','🃗':'7','🃘':'8','🃙':'9','🃚':'T','🃛':'J','🃝':'Q','🃞':'K','🂺':'T','🃚':'T','🃊':'T','🂪':'T'}
def parse_hand(hand_str):
 ranks_chars=[];suits=[]
 for card_char in hand_str:
  if card_char in UNICODE_MAP:
   ranks_chars.append(UNICODE_MAP[card_char])
   code_point=ord(card_char)
   if 0x1F0A0<=code_point<=0x1F0AF:suits.append('♠')
   elif 0x1F0B0<=code_point<=0x1F0BF:suits.append('♥')
   elif 0x1F0C0<=code_point<=0x1F0CF:suits.append('♦')
   elif 0x1F0D0<=code_point<=0x1F0DF:suits.append('♣')
  else:return[],[]
 ranks_values=[]
 for char in ranks_chars:
  if char=='A':ranks_values.append(14)
  elif char=='K':ranks_values.append(13)
  elif char=='Q':ranks_values.append(12)
  elif char=='J':ranks_values.append(11)
  elif char=='T':ranks_values.append(10)
  else:ranks_values.append(int(char))
 return ranks_values,suits
def get_hand_type(hand_str):
 ranks,suits=parse_hand(hand_str)
 if len(ranks)!=5:return"Invalid Hand"
 ranks.sort()
 is_flush=len(set(suits))==1
 is_straight=False
 if ranks==[2,3,4,5,14]:is_straight=True;ranks_for_straight=[1,2,3,4,5]
 elif len(set(ranks))==5 and ranks[4]-ranks[0]==4:is_straight=True;ranks_for_straight=ranks
 else:ranks_for_straight=ranks
 rank_counts=Counter(ranks);counts=sorted(rank_counts.values(),reverse=True)
 if is_flush and is_straight:
  if ranks_for_straight==[10,11,12,13,14]:return"Royal Flush"
  else:return"Straight Flush"
 elif counts==[4,1]:return"Four of a Kind"
 elif counts==[3,2]:return"Full House"
 elif is_flush:return"Flush"
 elif is_straight:return"Straight"
 elif counts==[3,1,1]:return"Three of a Kind"
 elif counts==[2,2,1]:return"Two Pair"
 elif counts==[2,1,1,1]:return"Pair"
 else:return"High Card"
def main():
 if len(sys.argv)<2:print("Usage: python script.py \"hand1\" \"hand2\" ...");sys.exit(1)
 for hand_str in sys.argv[1:]:print(get_hand_type(hand_str))
if __name__=='__main__':main()
```

---

## 76. polyominoes
```python
import sys
from collections import deque
def normalize_polyomino(polyomino):
 min_x=min(x for x,y in polyomino)
 min_y=min(y for x,y in polyomino)
 return frozenset((x-min_x,y-min_y)for x,y in polyomino)
def get_all_rotations(polyomino):
 rotations=set()
 current=polyomino
 for _ in range(4):
  rotations.add(current)
  current=frozenset((-y,x)for x,y in current)
 return rotations
def print_polyomino(polyomino):
 if not polyomino:return
 min_x=min(x for x,y in polyomino)
 max_x=max(x for x,y in polyomino)
 min_y=min(y for x,y in polyomino)
 max_y=max(y for x,y in polyomino)
 width=max_x-min_x+1
 height=max_y-min_y+1
 grid=[[' 'for _ in range(width)]for _ in range(height)]
 for x,y in polyomino:grid[y-min_y][x-min_x]='#'
 for row in grid:print(''.join(row))
def main():
 sys.setrecursionlimit(2000)
 all_polyominoes=[[],[],[],[],[],[],[]]
 polyomino_set={frozenset([(0,0)])}
 all_polyominoes[1]=polyomino_set
 for size in range(2,7):
  new_polyominoes=set()
  for polyomino in all_polyominoes[size-1]:
   for x,y in polyomino:
    for dx,dy in[(0,1),(0,-1),(1,0),(-1,0)]:
     new_square=(x+dx,y+dy)
     if new_square not in polyomino:
      new_polyomino=polyomino.union({new_square})
      new_polyominoes.add(normalize_polyomino(new_polyomino))
  all_polyominoes[size]=new_polyominoes
 for size in range(1,7):
  printed_polys=set()
  for canonical_polyomino in all_polyominoes[size]:
   all_rotations=get_all_rotations(canonical_polyomino)
   for rotation in all_rotations:
    normalized_rotation=normalize_polyomino(rotation)
    if normalized_rotation not in printed_polys:
     print_polyomino(normalized_rotation)
     print()
     printed_polys.add(normalized_rotation)
if __name__=='__main__':main()
```

---

## 77. prime-numbers
```python
import math

def is_prime(n):
  """
  Checks if a number is prime.

  Args:
    n: The number to check.

  Returns:
    True if the number is prime, False otherwise.
  """
  if n <= 1:
    return False
  for i in range(2, int(math.sqrt(n)) + 1):
    if n % i == 0:
      return False
  return True

def print_primes(limit):
  """
  Prints all prime numbers from 1 to limit (inclusive), each on a new line.

  Args:
    limit: The upper limit of the range.
  """
  for i in range(2, limit + 1):
    if is_prime(i):
      print(i)

if __name__ == "__main__":
  print_primes(100)
```

---

## 78. prime-numbers-long
```python
def is_prime(n):
    """
    Check if a number is prime.

    Args:
        n: The number to check.

    Returns:
        True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def print_primes(limit):
    """
    Print all prime numbers from 1 to limit inclusive, each on their own line.

    Args:
        limit: The upper limit for prime number generation.
    """
    for i in range(2, limit + 1):
        if is_prime(i):
            print(i)

if __name__ == "__main__":
    print_primes(10000)
```

---

## 79. proximity-grid
```python
import sys
from collections import deque
BASE62_CHARS="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
def to_base62(n):
 if n==0:return BASE62_CHARS[0]
 s=""
 while n>0:s=BASE62_CHARS[n%62]+s;n//=62
 return s
def solve_grid(grid_input):
 rows,cols=9,9
 grid=[list(row)for row in grid_input.split('\n')]
 distances=[[-1]*cols for _ in range(rows)]
 queue=deque()
 for r in range(rows):
  for c in range(cols):
   if grid[r][c]=='0':distances[r][c]=0;queue.append((r,c))
 while queue:
  r,c=queue.popleft()
  for dr,dc in[(0,1),(0,-1),(1,0),(-1,0)]:
   nr,nc=r+dr,c+dc
   if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]=='-' and distances[nr][nc]==-1:
    distances[nr][nc]=distances[r][c]+1
    queue.append((nr,nc))
 output_grid=[list(row)for row in grid_input.split('\n')]
 for r in range(rows):
  for c in range(cols):
   if distances[r][c]>0:output_grid[r][c]=to_base62(distances[r][c])
 return["".join(row)for row in output_grid]
def main():
 if len(sys.argv)<2:print("Usage: python script.py \"grid1\" \"grid2\" ...");sys.exit(1)
 for i,grid_arg in enumerate(sys.argv[1:]):
  if i>0:print()
  result_grid=solve_grid(grid_arg)
  for line in result_grid:print(line)
if __name__=='__main__':main()
```

---

## 80. qr-decoder(Error)
```python
def decode_qr(qr_code):
    """Decodes a Version-1 QR code from an ASCII-art representation."""

    def extract_bits(qr_code):
        """Extracts bits from the QR code based on 'vv' and '^^' patterns."""
        bits = []
        height = len(qr_code)
        width = len(qr_code[0])

        for strip_index in range(10):
            # Process each strip from right to left
            x = width - 8 - strip_index  # Calculate x-coordinate

            # Zig-zag upwards (^^)
            for y in range(height - 1, -1, -1):  # Start from the bottom
                if qr_code[y][x] == '^':
                    bit = 1 if qr_code[y][x] == '#' else 0
                    if (x + y) % 2 == 0:
                        bit = 1 - bit  # Invert the bit
                    bits.append(bit)

            # Zig-zag downwards (vv)
            for y in range(0, height):  # Start from the top
                if qr_code[y][x] == 'v':
                    bit = 1 if qr_code[y][x] == '#' else 0
                    if (x + y) % 2 == 0:
                        bit = 1 - bit  # Invert the bit
                    bits.append(bit)

        return bits

    def bits_to_bytes(bits):
        """Converts a list of bits to a list of bytes."""
        byte_string = ""
        for bit in bits:
            byte_string += str(bit)

        byte_list = []
        for i in range(0, len(byte_string), 8):
            byte = byte_string[i:i+8]
            byte_list.append(int(byte, 2))
        return byte_list

    # Extract bits
    bits = extract_bits(qr_code)

    # Convert bits to bytes
    byte_list = bits_to_bytes(bits)
    
    # Extract relevant information
    encoding_type = bin(byte_list[0])[2:].zfill(8)[:4] #First byte contains encoding and length
    length = bin(byte_list[0])[2:].zfill(8)[4:] + bin(byte_list[1])[2:].zfill(8)[2:]
    
    # Extract the 17 bytes message.
    message_bytes = byte_list[2:19]
    
    #Convert bytes to chars and return the message
    message = ''.join(chr(byte) for byte in message_bytes)
    
    return message

# Example Usage:
qr_code_data = [
    "#######  vv^^ #######",
    "#     #  vv^^ #     #",
    "# ### # #vv^^ # ### #",
    "# ### #  vv^^ # ### #",
    "# ### #  vv^^ # ### #",
    "#     #  vv^^ #     #",
    "####### # # # #######",
    "#vv^^                ",
    "### #####vv^^##   #",
    "vv^^vv ^^vv^^vv^^vv^^",
    "vv^^vv#^^vv^^vv^^vv^^",
    "vv^^vv ^^vv^^vv^^vv^^",
    "vv^^vv#^^vv^^vv^^vv^^",
    "#vv^^vv^^vv^^        ",
    "####### #vv^^vv^^vv^^",
    "#     # #vv^^vv^^vv^^",
    "# ### # #vv^^vv^^vv^^",
    "# ### #  vv^^vv^^vv^^",
    "# ### # #vv^^vv^^vv^^",
    "#     # #vv^^vv^^vv^^",
    "####### #vv^^vv^^vv^^"
]

# Replace 'v' and '^' with '#' and ' ' for proper bit extraction.
modified_qr_code = []
for row in qr_code_data:
    modified_row = row.replace('v', '#').replace('^', ' ')
    modified_qr_code.append(modified_row)

decoded_message = decode_qr(modified_qr_code)
print(decoded_message)
```

---

## 81. quine
```python
s = 's = %r
print(s %% s)'
print(s % s)

```

---

## 82. recamán
```python
def generate_sequence(length=250):
    """
    Generates a sequence based on the described rules and prints the first 'length' terms.

    Args:
        length: The number of terms to generate.  Defaults to 250.
    """

    sequence = [0]
    generated_numbers = {0}  # Use a set for efficient membership testing

    for n in range(1, length):
        minus_n = sequence[-1] - n
        if minus_n > 0 and minus_n not in generated_numbers:
            sequence.append(minus_n)
            generated_numbers.add(minus_n)
        else:
            plus_n = sequence[-1] + n
            sequence.append(plus_n)
            generated_numbers.add(plus_n)

    for term in sequence:
        print(term)

if __name__ == "__main__":
    generate_sequence()
```

---

## 83. repeating-decimals
```python
import sys
def get_decimal_expansion(numerator,denominator):
 if denominator==0:return"Error: Division by zero"
 sign=""
 if(numerator<0)!=(denominator<0):sign="-"
 numerator=abs(numerator)
 denominator=abs(denominator)
 integer_part=str(numerator//denominator)
 remainder=numerator%denominator
 if remainder==0:return sign+integer_part
 seen_remainders={}
 decimal_parts=[]
 index=0
 while remainder!=0 and remainder not in seen_remainders:
  seen_remainders[remainder]=index
  remainder*=10
  decimal_parts.append(str(remainder//denominator))
  remainder%=denominator
  index+=1
 if remainder==0:return sign+integer_part+"."+"".join(decimal_parts)
 else:
  start_of_repeat=seen_remainders[remainder]
  non_repeating="".join(decimal_parts[:start_of_repeat])
  repeating="".join(decimal_parts[start_of_repeat:])
  return f"{sign}{integer_part}.{non_repeating}({repeating})"
def main():
 if len(sys.argv)<2:print("Usage: python script.py a/b c/d ...");sys.exit(1)
 for arg in sys.argv[1:]:
  try:
   numerator_str,denominator_str=arg.split('/')
   numerator=int(numerator_str)
   denominator=int(denominator_str)
   result=get_decimal_expansion(numerator,denominator)
   print(result)
  except ValueError:print(f"Error: Invalid argument format '{arg}'")
  except IndexError:print(f"Error: Invalid argument format '{arg}'")
if __name__=='__main__':main()
```

---

## 84. reverse-polish-notation
```python
import sys
def evaluate_rpn(expression):
 stack=[]
 tokens=expression.split()
 for token in tokens:
  if token.isdigit():stack.append(int(token))
  else:
   operand2=stack.pop()
   operand1=stack.pop()
   if token=='+':result=operand1+operand2
   elif token=='-':result=operand1-operand2
   elif token=='*':result=operand1*operand2
   elif token=='/':result=operand1//operand2
   else:raise ValueError(f"Unknown operator: {token}")
   stack.append(result)
 if len(stack)!=1:raise ValueError("Invalid RPN expression: more than one value left on stack.")
 return stack.pop()
def main():
 if len(sys.argv)<2:print("Usage: python script.py '3 2 +' '10 5 / 2 *'");sys.exit(1)
 for arg in sys.argv[1:]:
  try:
   result=evaluate_rpn(arg)
   print(result)
  except(IndexError,ValueError)as e:print(f"Error evaluating '{arg}': {e}",file=sys.stderr)
if __name__=='__main__':main()
```

---

## 85. reversi
```python
import sys
DIRS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def legal_moves(board):
 n=8
 out=[list(row)for row in board]
 for y in range(n):
  for x in range(n):
   if board[y][x]!='.':continue
   for dy,dx in DIRS:
    ny,nx=y+dy,x+dx
    cnt=0
    while 0<=ny<n and 0<=nx<n and board[ny][nx]=='X':
     cnt+=1
     ny+=dy;nx+=dx
    if cnt>0 and 0<=ny<n and 0<=nx<n and board[ny][nx]=='O':
     out[y][x]='!'
     break
 return["".join(row)for row in out]
def main():
 args=sys.argv[1:]
 results=[]
 for arg in args:
  lines=arg.splitlines()
  if len(lines)!=8:continue
  board=[line.rstrip()for line in lines]
  results.extend(legal_moves(board))
  results.append("")
 if results and results[-1]=="":results.pop()
 print("\n".join(results))
if __name__=='__main__':main()
```

---

## 86. rijndael-s-box
```python
sbox=[0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16]
for i in range(0,256,16):print(' '.join(f'{b:02x}'for b in sbox[i:i+16]))
```

---

## 87. rock-paper-scissors-spock-lizard
```python
import sys
rules={'✂':{'📄':'cuts','🦎':'decapitates'},'📄':{'💎':'covers','🖖':'disproves'},'💎':{'✂':'crushes','🦎':'crushes'},'🦎':{'📄':'eats','🖖':'poisons'},'🖖':{'✂':'smashes','💎':'vaporizes'}}
for arg in sys.argv[1:]:
 a,b=arg[0],arg[1]
 if a==b:print("Tie")
 elif b in rules.get(a,{}):print(f"{a} {rules[a][b]} {b}")
 else:print(f"{b} {rules[b][a]} {a}")
```

---

## 88. roman-to-arabic
```python
import sys
roman={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
for s in sys.argv[1:]:
 total=0
 for i in range(len(s)):
  val=roman[s[i]]
  if i+1<len(s)and roman[s[i]]<roman[s[i+1]]:total-=val
  else:total+=val
 print(total)
```

---

## 89. rot13
```python
import sys
for s in sys.argv[1:]:print(''.join(chr((ord(c)-65+13)%26+65)if c.isupper()else chr((ord(c)-97+13)%26+97)if c.islower()else c for c in s))
```

---

## 90. rule-110
```python
w=199
a=[0]*w
a[w//2]=1
for _ in range(100):
 print(''.join('█'if c else' 'for c in a))
 b=[0]+a+[0]
 a=[int(b[i-1:i+2]in([1,1,0],[1,0,1],[0,1,1],[0,1,0],[0,0,1]))for i in range(1,w+1)]
```

---

## 91. scrambled-sort
```python
import sys
for arg in sys.argv[1:]:
 parts=arg.split()
 alphabet,words=parts[0],parts[1:]
 order={c:i for i,c in enumerate(alphabet)}
 print(' '.join(sorted(words,key=lambda word:[order[c]for c in word])))
```

---

## 92. set
```python
import sys
from itertools import combinations
def is_set(a,b,c):
 for i in range(4):
  s={a[i],b[i],c[i]}
  if len(s)==2:return False
 return True
for group in sys.argv[1:]:
 cards=group.split()
 for i,j,k in combinations(range(12),3):
  if is_set(cards[i],cards[j],cards[k]):print(cards[i],cards[j],cards[k]);break
```

---

## 93. seven-segment
```python
import sys
digits={'0':[' _ ','| |','|_|'],'1':['   ','  |','  |'],'2':[' _ ',' _|','|_ '],'3':[' _ ',' _|',' _|'],'4':['   ','|_|','  |'],'5':[' _ ','|_ ',' _|'],'6':[' _ ','|_ ','|_|'],'7':[' _ ','  |','  |'],'8':[' _ ','|_|','|_|'],'9':[' _ ','|_|',' _|']}
for arg in sys.argv[1:]:
 lines=['','','']
 for ch in arg:
  for i in range(3):lines[i]+=digits[ch][i]
 print('\n'.join(lines))
```

---

## 94. si-units(Error)
```python
def si_unit_to_base_units(unit_string):
    """
    Converts an SI unit with a possible prefix to a product of powers of ten and its base units.

    Args:
        unit_string: The SI unit with a possible prefix (e.g., GW, Mrad, kg).

    Returns:
        A string representing the unit as a product of a power of ten and its base units
        (e.g., "10^9 kg m^2 s^-3", "10^6").
    """

    prefixes = {
        'Q': 30, 'R': 27, 'Y': 24, 'Z': 21, 'E': 18, 'P': 15,
        'T': 12, 'G': 9, 'M': 6, 'k': 3, 'h': 2, 'da': 1,
        'd': -1, 'c': -2, 'm': -3, 'μ': -6, 'n': -9, 'p': -12,
        'f': -15, 'a': -18, 'z': -21, 'y': -24, 'r': -27, 'q': -30
    }

    units = {
        'rad': {'base_units': '1', 'power': 0},
        'sr': {'base_units': '1', 'power': 0},
        'Hz': {'base_units': 's^-1', 'power': 0},
        'N': {'base_units': 'kg m s^-2', 'power': 0},
        'Pa': {'base_units': 'kg m^-1 s^-2', 'power': 0},
        'J': {'base_units': 'kg m^2 s^-2', 'power': 0},
        'W': {'base_units': 'kg m^2 s^-3', 'power': 0},
        'C': {'base_units': 'A s', 'power': 0},
        'V': {'base_units': 'kg m^2 s^-3 A^-1', 'power': 0},
        'F': {'base_units': 'kg^-1 m^-2 s^4 A^2', 'power': 0},
        'Ω': {'base_units': 'kg m^2 s^-3 A^-2', 'power': 0},
        'S': {'base_units': 'kg^-1 m^-2 s^3 A^2', 'power': 0},
        'Wb': {'base_units': 'kg m^2 s^-2 A^-1', 'power': 0},
        'T': {'base_units': 'kg s^-2 A^-1', 'power': 0},
        'H': {'base_units': 'kg m^2 s^-2 A^-2', 'power': 0},
        '°C': {'base_units': 'K', 'power': 0},
        'lm': {'base_units': 'cd', 'power': 0},
        'lx': {'base_units': 'cd m^-2', 'power': 0},
        'Bq': {'base_units': 's^-1', 'power': 0},
        'Gy': {'base_units': 'm^2 s^-2', 'power': 0},
        'Sv': {'base_units': 'm^2 s^-2', 'power': 0},
        'kat': {'base_units': 'mol s^-1', 'power': 0},
        'kg': {'base_units': 'kg', 'power': 0},
        'g': {'base_units': 'kg', 'power': -3}  # Special case: gram
    }

    unit = unit_string
    power = 0

    # Handle prefixes
    for prefix, pwr in prefixes.items():
        if unit.startswith(prefix):
            unit = unit[len(prefix):]
            power = pwr
            break

    # Special handling for kg: it already has a prefix
    if unit == 'kg':
        base_units = units['kg']['base_units']
        unit_power = 0

    #Special handling for g and prefixes for g
    elif unit == 'g':
        base_units = units['g']['base_units']
        unit_power = units['g']['power'] + power
        power = 0

    elif unit in units:
        base_units = units[unit]['base_units']
        unit_power = units[unit]['power']
    else:
        return "Invalid unit" # Added error handling

    total_power = power + unit_power

    # Format the output
    if units.get(unit_string.lstrip('QMTRYZEPTGkhdamunpfazyrq'),{}).get('base_units','') == '1': #check unit
        if total_power == 0:
            return "1"
        elif total_power == 1:
            return "10"
        else:
            return f"10^{total_power}"
    else:
        if total_power == 0:
            power_string = "1"
        elif total_power == 1:
            power_string = "10"
        else:
            power_string = f"10^{total_power}"

        return f"{power_string} {base_units}"
```

---

## 95. serpiński-triangle
```python
def sierpinski_triangle(n):
 if n==0:return["▲"]
 prev_triangle=sierpinski_triangle(n-1)
 prev_size=len(prev_triangle)
 current_triangle=[]
 for row in prev_triangle:current_triangle.append(" "*prev_size+row+" "*prev_size)
 for row in prev_triangle:current_triangle.append(row+" "+row)
 return current_triangle
def main():
 order=4
 triangle_rows=sierpinski_triangle(order)
 for row in triangle_rows:print(row)
if __name__=='__main__':main()
```

---

## 96. smith-numbers
```python
def sum_digits(n):
 s=0
 while n:s+=n%10;n//=10
 return s
def get_prime_factors(n):
 factors=[];d=2;temp=n
 while d*d<=temp:
  while temp%d==0:factors.append(d);temp//=d
  d+=1
 if temp>1:factors.append(temp)
 return factors
def is_prime(n):
 if n<=1:return False
 if n<=3:return True
 if n%2==0 or n%3==0:return False
 i=5
 while i*i<=n:
  if n%i==0 or n%(i+2)==0:return False
  i+=6
 return True
def find_smith_numbers(limit):
 smith_numbers=[]
 for num in range(4,limit+1):
  if not is_prime(num):
   digit_sum=sum_digits(num)
   prime_factors=get_prime_factors(num)
   prime_factors_digit_sum=0
   for factor in prime_factors:prime_factors_digit_sum+=sum_digits(factor)
   if digit_sum==prime_factors_digit_sum:smith_numbers.append(num)
 return smith_numbers
if __name__=='__main__':
 limit=10000
 smith_numbers=find_smith_numbers(limit)
 for num in smith_numbers:print(num)
```

---

## 97. spelling-numbers
```python
import sys
def spell_number(n):
 ones=["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
 tens=["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
 if n==0:return"zero"
 if n==1000:return"one thousand"
 result=[]
 if n>=100:
  result.append(ones[n//100])
  result.append("hundred")
  if n%100!=0:result.append("and")
  n%=100
 if n>0:
  if n<20:result.append(ones[n])
  else:
   result.append(tens[n//10])
   if n%10!=0:result.append(ones[n%10])
 output=" ".join(result).replace("ty ","ty-")
 return output
def main():
 if len(sys.argv)<2:print("Usage: python script.py <integer1> <integer2> ...");sys.exit(1)
 for arg in sys.argv[1:]:
  try:
   num=int(arg)
   if 0<=num<=1000:print(spell_number(num))
   else:print(f"Error: {num} is outside the range of 0 to 1000.",file=sys.stderr)
  except ValueError:print(f"Error: '{arg}' is not a valid integer.",file=sys.stderr)
if __name__=='__main__':main()
```

---

## 98. star-wars-gpt
```python
import sys
from collections import Counter
def predict_word(corpus_text,prompt):
 corpus=corpus_text.split()
 if not prompt or prompt not in corpus:return""
 following_words=[corpus[i+1]for i,word in enumerate(corpus[:-1])if word==prompt]
 if not following_words:return""
 counts=Counter(following_words)
 max_count=max(counts.values())
 modes={word for word,count in counts.items()if count==max_count}
 for word in following_words:
  if word in modes:return word
def main():
 if len(sys.argv)<2:return
 for arg in sys.argv[1:]:
  lines=arg.strip().split('\n')
  if len(lines)>1:
   corpus=lines[0]
   prompts=lines[1:]
   for prompt in prompts:print(predict_word(corpus,prompt))
if __name__=='__main__':main()
```

---

## 99. star-wars-opening-crawl
```python
import sys
def justify_line(words,width):
 num_words=len(words)
 if num_words<=1:return" ".join(words)
 total_word_length=sum(len(w)for w in words)
 total_spaces=width-total_word_length
 num_gaps=num_words-1
 if num_gaps<=0:return"".join(words)
 base_spaces=total_spaces//num_gaps
 extra_spaces=total_spaces%num_gaps
 parts=[]
 for i,word in enumerate(words):
  parts.append(word)
  if i<num_gaps:parts.append(" "*(base_spaces+(1 if i<extra_spaces else 0)))
 return"".join(parts)
def format_crawl(input_text):
 lines=input_text.strip().split('\n')
 if not lines:return
 try:initial_I,initial_W=map(int,lines[0].split())
 except(ValueError,IndexError):return
 current_I=initial_I
 current_W=initial_W
 line_counter=0
 formatted_output=[]
 paragraphs=lines[1:]
 for paragraph in paragraphs:
  all_words=paragraph.split()
  word_idx=0
  while word_idx<len(all_words):
   current_line_words=[]
   current_line_length=0
   start_word_idx=word_idx
   while word_idx<len(all_words):
    word=all_words[word_idx]
    word_len=len(word)
    if not current_line_words:
     current_line_words.append(word)
     current_line_length=word_len
    elif current_line_length+1+word_len<=current_W:
     current_line_words.append(word)
     current_line_length+=1+word_len
    else:break
    word_idx+=1
   is_last_line_of_paragraph=(word_idx==len(all_words))
   prefix=" "*current_I
   if is_last_line_of_paragraph:formatted_line=" ".join(current_line_words)
   else:formatted_line=justify_line(current_line_words,current_W)
   formatted_output.append(prefix+formatted_line)
   line_counter+=1
   if line_counter%2==0:
    current_I-=1
    current_W+=2
  if paragraph!=paragraphs[-1]:
   formatted_output.append("")
   line_counter+=1
   if line_counter%2==0:
    current_I-=1
    current_W+=2
 return"\n".join(formatted_output)
def main():
 if len(sys.argv)<2:
  print("Usage: python script.py \"<test_case_1>\" \"<test_case_2>\" ...",file=sys.stderr)
  sys.exit(1)
 all_outputs=[]
 for arg in sys.argv[1:]:
  formatted_text=format_crawl(arg)
  if formatted_text:all_outputs.append(formatted_text)
 print("\n\n".join(all_outputs))
if __name__=='__main__':main()
```

---

## 100. sudoku
```python
import sys
def solve_sudoku(board):
 empty_cell=find_empty_cell(board)
 if not empty_cell:return True
 row,col=empty_cell
 for num in range(1,10):
  if is_valid(board,row,col,str(num)):
   board[row][col]=str(num)
   if solve_sudoku(board):return True
   board[row][col]='.'
 return False
def find_empty_cell(board):
 for row in range(9):
  for col in range(9):
   if board[row][col]=='.':return(row,col)
 return None
def is_valid(board,row,col,num):
 for i in range(9):
  if board[row][i]==num:return False
 for i in range(9):
  if board[i][col]==num:return False
 start_row=(row//3)*3
 start_col=(col//3)*3
 for i in range(3):
  for j in range(3):
   if board[start_row+i][start_col+j]==num:return False
 return True
def print_formatted_board(board):
 top_border="┏━━━┯━━━┯━━━┳━━━┯━━━┯━━━┳━━━┯━━━┯━━━┓"
 middle_border="┠───┼───┼───╂───┼───┼───╂───┼───┼───┨"
 heavy_border="┣━━━┿━━━┿━━━╋━━━┿━━━┿━━━╋━━━┿━━━┿━━━┫"
 bottom_border="┗━━━┷━━━┷━━━┻━━━┷━━━┷━━━┻━━━┷━━━┷━━━┛"
 print(top_border)
 for i in range(9):
  row_str="┃"
  for j in range(9):
   row_str+=f" {board[i][j]} "
   if(j+1)%3==0:row_str+="┃"
   else:row_str+="│"
  print(row_str)
  if i==8:print(bottom_border)
  elif(i+1)%3==0:print(heavy_border)
  else:print(middle_border)
def main():
 if len(sys.argv)!=10:
  print("Usage: python script.py <row1> <row2> ... <row9>",file=sys.stderr)
  sys.exit(1)
 board=[]
 for i in range(1,10):
  row_str=sys.argv[i]
  if len(row_str)!=9:
   print(f"Error: Row {i} must have 9 characters.",file=sys.stderr)
   sys.exit(1)
  board.append(list(row_str.replace('_','.')))
 if solve_sudoku(board):print_formatted_board(board)
 else:print("该数独无解。")
if __name__=='__main__':main()
```

---

## 101. sudoku-fill-in
```python
import sys
def parse_formatted_board(input_string):
 lines=input_string.strip().split('\n')
 board=[]
 for line in lines:
  if"┃"in line:
   row=[line[2+i*4]if not line[2+i*4].isspace()else'.'for i in range(9)]
   board.append(row)
 if len(board)!=9 or any(len(r)!=9 for r in board):raise ValueError("输入格式不正确，无法解析为9x9数独棋盘。")
 return board
def solve_sudoku(board):
 empty_cell=find_empty_cell(board)
 if not empty_cell:return True
 row,col=empty_cell
 for num in range(1,10):
  num_str=str(num)
  if is_valid(board,row,col,num_str):
   board[row][col]=num_str
   if solve_sudoku(board):return True
   board[row][col]='.'
 return False
def find_empty_cell(board):
 for r,row in enumerate(board):
  for c,cell in enumerate(row):
   if cell=='.':return r,c
 return None
def is_valid(board,row,col,num_str):
 if num_str in board[row]or num_str in[board[i][col]for i in range(9)]:return False
 start_row,start_col=(row//3)*3,(col//3)*3
 for i in range(3):
  for j in range(3):
   if board[start_row+i][start_col+j]==num_str:return False
 return True
def print_formatted_board(board):
 top_border="┏━━━┯━━━┯━━━┳━━━┯━━━┯━━━┳━━━┯━━━┯━━━┓"
 middle_border="┠───┼───┼───╂───┼───┼───╂───┼───┼───┨"
 heavy_border="┣━━━┿━━━┿━━━╋━━━┿━━━┿━━━╋━━━┿━━━┿━━━┫"
 bottom_border="┗━━━┷━━━┷━━━┻━━━┷━━━┷━━━┻━━━┷━━━┷━━━┛"
 print(top_border)
 for i,row in enumerate(board):
  row_str="┃"+"".join(f" {cell} "+("┃"if(j+1)%3==0 else"│")for j,cell in enumerate(row))
  print(row_str)
  if i==8:print(bottom_border)
  elif(i+1)%3==0:print(heavy_border)
  else:print(middle_border)
def main():
 if len(sys.argv)<2:
  print("Usage: python script.py \"<formatted_sudoku_board>\"",file=sys.stderr)
  sys.exit(1)
 try:
  board=parse_formatted_board(sys.argv[1])
  if solve_sudoku(board):print_formatted_board(board)
  else:print("The puzzle has no solution.")
 except ValueError as e:print(e,file=sys.stderr)
if __name__=='__main__':main()
```

---

## 102. ten-pin-bowling
```python
import sys
def calculate_score(game_string):
 game_string=game_string.replace('⑧','8').replace('⑦','7').replace('⑥','6').replace('⑤','5')
 char_to_score={'X':10,'/':-1,'F':0,'-':0}
 rolls=[]
 for char in game_string:
  if char==' ':continue
  elif char in char_to_score:rolls.append(char_to_score[char])
  elif char.isdigit():rolls.append(int(char))
 for i,score in enumerate(rolls):
  if score==-1:rolls[i]=10-rolls[i-1]
 total_score,roll_index=0,0
 for frame in range(10):
  if frame<9:
   if rolls[roll_index]==10:
    total_score+=10+rolls[roll_index+1]+rolls[roll_index+2]
    roll_index+=1
   elif rolls[roll_index]+rolls[roll_index+1]==10:
    total_score+=10+rolls[roll_index+2]
    roll_index+=2
   else:
    total_score+=rolls[roll_index]+rolls[roll_index+1]
    roll_index+=2
  else:total_score+=sum(rolls[roll_index:])
 return total_score
def main():
 if len(sys.argv)<2:return
 game_strings=sys.argv[1:]
 for game_string in game_strings:print(calculate_score(game_string))
if __name__=='__main__':main()
```

---

## 103. tic-tac-toe
```python
import sys
def check_winner(board_string):
 board=[list(row)for row in board_string.split('\n')]
 for i in range(3):
  if board[i][0]==board[i][1]==board[i][2]!='.':return board[i][0]
  if board[0][i]==board[1][i]==board[2][i]!='.':return board[0][i]
 if board[0][0]==board[1][1]==board[2][2]!='.':return board[0][0]
 if board[0][2]==board[1][1]==board[2][0]!='.':return board[0][2]
 return'-'
def main():
 if len(sys.argv)<2:return
 for board_arg in sys.argv[1:]:print(check_winner(board_arg))
if __name__=='__main__':main()
```

---

## 104. time-distance
```python
import sys
def format_time_distance(seconds):
 if seconds==0:return"now"
 is_future=seconds>0
 abs_seconds=abs(seconds)
 units=[('year',31536000),('month',2592000),('week',604800),('day',86400),('hour',3600),('minute',60),('second',1)]
 unit_name=''
 quantity=0
 for name,unit_seconds in units:
  if abs_seconds>=unit_seconds:
   quantity=abs_seconds//unit_seconds
   unit_name=name
   break
 else:
  quantity=abs_seconds
  unit_name='second'
 if quantity==1:
  prefix="an"if unit_name=="hour"else"a"
  formatted_string=f"{prefix} {unit_name}"
 else:formatted_string=f"{quantity} {unit_name}s"
 return f"in {formatted_string}"if is_future else f"{formatted_string} ago"
def main():
 if len(sys.argv)<2:return
 for time_str in sys.argv[1:]:
  time_int=int(time_str)
  print(format_time_distance(time_int))
if __name__=='__main__':main()
```

---

## 105. tongue-twisters
```python
def print_tongue_twisters():
    tongue_twisters = [
        """How much wood would a woodchuck chuck,
If a woodchuck could chuck wood?
A woodchuck would chuck all the wood he could chuck
If a woodchuck would chuck wood.""",
        """Peter Piper picked a peck of pickled peppers.
A peck of pickled peppers Peter Piper picked.
If Peter Piper picked a peck of pickled peppers,
Where's the peck of pickled peppers Peter Piper picked?""",
        """She sells seashells by the seashore,
The shells she sells are seashells, I'm sure.
So if she sells seashells on the seashore,
Then I'm sure she sells seashore shells."""
    ]

    for i, twister in enumerate(tongue_twisters):
        print(twister)
        if i < len(tongue_twisters) - 1:
            print()

if __name__ == "__main__":
    print_tongue_twisters()
```

---

## 106. topological-sort
```python
import sys
def topological_sort(edges_string):
 edges=[list(map(int,line.split()))for line in edges_string.strip().split('\n')if line.strip()]
 if not edges:return""
 nodes={n for edge in edges for n in edge}
 num_nodes=max(nodes)+1 if nodes else 0
 adj={i:[]for i in range(num_nodes)}
 in_degree=[0]*num_nodes
 for u,v in edges:
  adj[u].append(v)
  in_degree[v]+=1
 queue=[i for i in range(num_nodes)if in_degree[i]==0]
 result=[]
 head=0
 while head<len(queue):
  u=queue[head]
  head+=1
  result.append(str(u))
  for v in adj[u]:
   in_degree[v]-=1
   if in_degree[v]==0:queue.append(v)
 return" ".join(result)if len(result)==num_nodes else"Error: A topological sort is not possible."
def main():
 if len(sys.argv)<2:return
 for arg in sys.argv[1:]:print(topological_sort(arg))
if __name__=='__main__':main()
```

---

## 107. transpose-sentence
```python
import sys
def transpose_sentence(sentence):
 words=sentence.split()
 if not words:return""
 max_len=max(len(word)for word in words)
 transposed_columns=[]
 for i in range(max_len):
  column_str=""
  for word in words:
   if i<len(word):column_str+=word[i]
  transposed_columns.append(column_str)
 return" ".join(transposed_columns)
def main():
 for sentence in sys.argv[1:]:print(transpose_sentence(sentence))
if __name__=='__main__':main()
```

---

## 108. united-states
```python
import sys
def main():
 state_abbreviations={"alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO","connecticut":"CT","delaware":"DE","district of columbia":"DC","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY"}
 if len(sys.argv)<2:
  print("Usage: python test.py \"State 1\" \"State 2\" ...",file=sys.stderr)
  sys.exit(1)
 for state_name in sys.argv[1:]:
  lower_name=state_name.lower()
  if lower_name in state_abbreviations:print(state_abbreviations[lower_name])
if __name__=='__main__':main()
```

---

## 109. vampire-numbers
```python
import sys
def find_vampire_numbers(num_digits):
 vampire_numbers=set()
 fang_digits=num_digits//2
 start_fang=10**(fang_digits-1)
 end_fang=10**fang_digits-1
 for a in range(start_fang,end_fang+1):
  for b in range(a,end_fang+1):
   if a%10==0 and b%10==0:continue
   product=a*b
   if len(str(product))!=num_digits:continue
   product_digits=sorted(str(product))
   fangs_digits=sorted(str(a)+str(b))
   if product_digits==fangs_digits:vampire_numbers.add(product)
 return vampire_numbers
def main():
 all_vampire_numbers=set()
 all_vampire_numbers.update(find_vampire_numbers(4))
 all_vampire_numbers.update(find_vampire_numbers(6))
 for number in sorted(all_vampire_numbers):print(number)
if __name__=='__main__':main()
```

---

## 110. van-eck-sequence
```python
def van_eck_sequence(n):
    """
    Generates the first n terms of the Van Eck sequence.

    Args:
      n: The number of terms to generate.

    Returns:
      A list containing the first n terms of the Van Eck sequence.
    """

    sequence = [0]
    seen = {0: 0}  # Store the last seen index of each number

    for i in range(1, n):
        last_term = sequence[-1]
        if last_term not in seen:
            next_term = 0
            sequence.append(next_term)
            seen[last_term] = i - 1 # store the index BEFORE adding current element, as this is last index for that term
            if next_term not in seen:
                seen[next_term] = i
        else:
            last_seen = seen[last_term]
            next_term = (i - 1) - last_seen  # Calculate the distance
            sequence.append(next_term)
            seen[last_term] = i - 1  # Update last seen index
            if next_term not in seen:
                seen[next_term] = i

    return sequence


if __name__ == "__main__":
    terms = van_eck_sequence(1000)
    for term in terms:
        print(term)
```

---

## 111. zeckendorf-representation
```python
import sys
def get_fibonacci_sequence(limit):
 fibs=[1,2]
 a,b=1,2
 while b<=limit:a,b=b,a+b;fibs.append(b)
 return fibs
def find_zeckendorf_representation(n,fibs):
 representation=[]
 for fib in reversed(fibs):
  if n>=fib:representation.append(str(fib));n-=fib
 return" + ".join(representation)
def main():
 fib_limit=2**31
 fibonacci_numbers=get_fibonacci_sequence(fib_limit)
 if len(sys.argv)<2:
  print("Usage: python script.py <number1> <number2> ...",file=sys.stderr)
  sys.exit(1)
 for arg in sys.argv[1:]:
  try:
   num=int(arg)
   if num<1 or num>=fib_limit:
    print(f"Input {num} is out of the valid range (1 to 2^31-1).",file=sys.stderr)
    continue
   representation=find_zeckendorf_representation(num,fibonacci_numbers)
   print(representation)
  except ValueError:print(f"Invalid input: '{arg}' is not a valid integer.",file=sys.stderr)
if __name__=='__main__':main()
```

---

## 112. zodiac-signs
```python
import sys
SUN_SIGNS=[('Capricorn','♑',(12,22),(12,31)),('Aquarius','♒',(1,20),(2,18)),('Pisces','♓',(2,19),(3,20)),('Aries','♈',(3,21),(4,19)),('Taurus','♉',(4,20),(5,20)),('Gemini','♊',(5,21),(6,21)),('Cancer','♋',(6,22),(7,22)),('Leo','♌',(7,23),(8,22)),('Virgo','♍',(8,23),(9,22)),('Libra','♎',(9,23),(10,22)),('Scorpio','♏',(10,23),(11,22)),('Sagittarius','♐',(11,23),(12,21)),('Capricorn','♑',(1,1),(1,19))]
SUN_ORDER=['♈','♉','♊','♋','♌','♍','♎','♏','♐','♑','♒','♓']
ASC_ROWS=[list('♒♓♈♉♊♋♌♍♎♏♐♑'),list('♓♈♉♊♋♌♍♎♏♐♑♒'),list('♈♉♊♋♌♍♎♏♐♑♒♓'),list('♉♊♋♌♍♎♏♐♑♒♓♈'),list('♊♋♌♍♎♏♐♑♒♓♈♉'),list('♋♌♍♎♏♐♑♒♓♈♉♊'),list('♌♍♎♏♐♑♒♓♈♉♊♋'),list('♍♎♏♐♑♒♓♈♉♊♋♌'),list('♎♏♐♑♒♓♈♉♊♋♌♍'),list('♏♐♑♒♓♈♉♊♋♌♍♎'),list('♐♑♒♓♈♉♊♋♌♍♎♏'),list('♑♒♓♈♉♊♋♌♍♎♏♐')]
def sun_sign(month,day):
 for name,sym,(m1,d1),(m2,d2)in SUN_SIGNS:
  if(month,day)>=(m1,d1)and(month,day)<=(m2,d2):return sym
 return''
def ascendant(sun_sym,hour):
 slot=hour//2
 row=ASC_ROWS[slot]
 idx=SUN_ORDER.index(sun_sym)
 return row[idx]
def main():
 for arg in sys.argv[1:]:
  date,time=arg.split()
  mm,dd=map(int,date.split('-'))
  hh,_=map(int,time.split(':'))
  sun=sun_sign(mm,dd)
  asc=ascendant(sun,hh)
  if asc!=sun:print(f"{sun}{asc}")
  else:print(sun)
if __name__=='__main__':main()
```

---

## 113. γ
```python
import decimal
from decimal import Decimal as D
decimal.getcontext().prec=1200
def bernoulli_numbers(m):
 A=[D(0)]*(2*m+1)
 B=[]
 for n in range(2*m+1):
  A[n]=D(1)/(n+1)
  for k in range(n,0,-1):A[k-1]=k*(A[k-1]-A[k])
  if n%2==0:B.append(A[0])
 return B
def compute_gamma():
 N=20000
 m=200
 H=sum(D(1)/D(k)for k in range(1,N+1))
 lnN=D(N).ln()
 B=bernoulli_numbers(m)
 corr=-D(1)/(2*D(N))
 for k in range(1,m+1):corr+=B[k]/(D(2*k)*(D(N)**(2*k)))
 return H-lnN+corr
γ=compute_gamma()
s=format(γ,'f')
intp,fracp=s.split('.')
fracp=(fracp+'0'*1000)[:1000]
print(f"{intp}.{fracp}")
```

---

## 114. λ
```python
POLY_COEFFS=[-6,3,-6,12,-4,7,-7,1,0,5,-2,-4,-12,2,7,12,-7,-10,-4,3,9,-7,0,-8,14,-3,9,2,-3,-10,-2,-6,1,10,-3,1,7,-7,7,-12,-5,8,6,10,-8,-8,-7,-3,9,1,6,6,-2,-3,-10,-2,3,5,2,-1,-1,-1,-1,-1,1,2,2,-1,-2,-1,0,1]
def eval_poly(x,coeffs,scale,deriv=False):
 n=len(coeffs)
 result=0
 if deriv:
  for i in range(n-1,0,-1):
   c=coeffs[i]*i
   result=(x*result)//scale+c*scale
 else:
  for i in range(n-1,-1,-1):
   c=coeffs[i]
   result=(x*result)//scale+c*scale
 return result
def compute_conway_constant():
 scale=10**2000
 x=4*scale//3
 for _ in range(20):
  f_val=eval_poly(x,POLY_COEFFS,scale,deriv=False)
  df_val=eval_poly(x,POLY_COEFFS,scale,deriv=True)
  if df_val==0:break
  x-=(f_val*scale)//df_val
 s=str(x)
 if len(s)<1001:s=s+"0"*(1001-len(s))
 return s[0]+"."+s[1:1001]
if __name__=='__main__':print(compute_conway_constant())
```

---

## 115. π
```python
import decimal
from decimal import Decimal as D,getcontext
def compute_pi(digits):
 getcontext().prec=digits+20
 C=426880*D(10005).sqrt()
 M,L,X,K,S=D(1),D(13591409),D(1),D(6),D(13591409)
 for k in range(1,digits//14+10):
  M=(M*(K**3-16*K))/(k**3)
  L+=545140134
  X*=-262537412640768000
  term=M*L/X
  S+=term
  if abs(term)<D(1)/(D(10)**(digits+5)):break
  K+=12
 return +C/S
def main():
 digits=1000
 pi=compute_pi(digits)
 s=format(pi,'f')
 intp,fracp=s.split('.')
 fracp=(fracp+'0'*digits)[:digits]
 print(f"{intp}.{fracp}")
if __name__=='__main__':main()
```

---

## 116. τ
```python
import decimal
from decimal import Decimal as D,getcontext
def compute_pi(digits):
 getcontext().prec=digits+20
 C=D(426880)*D(10005).sqrt()
 M,L,X,K,S=D(1),D(13591409),D(1),D(6),D(13591409)
 for k in range(1,digits//14+10):
  M=(M*(K**3-16*K))/(k**3)
  L+=D(545140134)
  X*=D(-262537412640768000)
  S+=M*L/X
  K+=D(12)
 return +C/S
def main():
 digits=1000
 pi=compute_pi(digits)
 tau=pi*D(2)
 s=format(tau,'f')
 intp,fracp=s.split('.')
 fracp=(fracp+'0'*digits)[:digits]
 print(f"{intp}.{fracp}")
if __name__=='__main__':main()
```

---

## 117. φ
```python
from decimal import Decimal, getcontext

def golden_ratio_digits(precision=1000):
    """
    Calculates and prints the first 'precision' decimal digits of the Golden Ratio (phi).

    Args:
        precision (int): The number of decimal digits to calculate.  Defaults to 1000.
    """

    getcontext().prec = precision + 10  # Add some extra precision to avoid rounding errors

    phi = (Decimal(5).sqrt() + 1) / 2

    phi_str = str(phi)

    # Find the position of the decimal point
    decimal_point_index = phi_str.find('.')

    # Extract digits after the decimal point, plus the leading digit
    digits = phi_str[:decimal_point_index + precision + 1]

    print(digits)


if __name__ == '__main__':
    golden_ratio_digits(1000)
```

---

## 118. √2
```python
import decimal
from decimal import Decimal as D,getcontext
def main():
 DIGITS=1000
 getcontext().prec=DIGITS+10
 two=D(2)
 root2=two.sqrt()
 s=format(root2,'f')
 int_part,frac_part=s.split('.')
 frac_part=(frac_part+'0'*DIGITS)[:DIGITS]
 print(f"{int_part}.{frac_part}")
if __name__=='__main__':main()
```

---

## 119. e
```python
import decimal
from decimal import Decimal as D,getcontext
def compute_e(digits):
 getcontext().prec=digits+20
 one,e_sum,term,k=D(1),D(1),D(1),1
 threshold=D(10)**(-(digits+5))
 while term>threshold:
  term/=D(k)
  e_sum+=term
  k+=1
 return +e_sum
def main():
 DIGITS=1000
 e=compute_e(DIGITS)
 s=format(e,'f')
 intp,fracp=s.split('.')
 fracp=(fracp+'0'*DIGITS)[:DIGITS]
 print(f"{intp}.{fracp}")
if __name__=='__main__':main()
```

---

