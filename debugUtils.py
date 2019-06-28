

if __name__ == "__main__":

    with open('[n]debug.log', 'w') as new_file:
        with open('./debug.log', 'r') as debugfile:

            [new_file.write('{}'.format(n)) for n in debugfile if 'ignite' not in n]
    print('Done')
