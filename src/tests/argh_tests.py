import argh

# declaring:

def echo(text):
    "Returns given word as is."
    return text
def greet(name, greeting='Hello'):
 "Greets the user with given name. The greeting is customizable."
 return greeting + ', ' + name

# assembling:
parser = argh.ArghParser()
parser.add_commands([echo, greet])

# dispatching:

if __name__ == '__main__':
    parser.dispatch()
argh.dispatch_command(main)

# adding help to `foo` which is in the function signature:
@arg('foo', help='blah')
# these are not in the signature so they go to **kwargs:
@arg('baz')
@arg('-q', '--quux')
# the function itself:
def cmd(foo, bar=1, *args, **kwargs):
    yield foo
    yield bar
    yield ', '.join(args)
    yield kwargs['baz']
    yield kwargs['quux']
