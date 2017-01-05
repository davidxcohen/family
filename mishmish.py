def drawPyramid(rows):
  result = ''

  for i in xrange(rows):
    print i
    row = ''
    row += ' ' * (rows - (i + 1))
    row += '*' * ((i + 1) * 2 - 1)

    result += row + '\n'

  return result