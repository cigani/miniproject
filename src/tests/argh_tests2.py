@arg('text', default='hello world', nargs='+', help='The message')
def echo(text):
    print text
