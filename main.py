# Image compression
#
# You'll need Python 2.7 and must install these packages:
#
#   scipy, numpy
#
# You can run this *only* on PNM images, which the netpbm library is used for.
#
# You can also display a PNM image using the netpbm library as, for example:
#
#   python netpbm.py images/cortex.pnm


import sys, os, math, time, netpbm
import numpy as np
import copy

# Text at the beginning of the compressed file, to identify it


headerText = 'my compressed image - v1.0'
twoBytes = 256*256


# Compress an image


def compress( inputFile, outputFile ):

  # Read the input file into a numpy array of 8-bit values
  #
  # The img.shape is a 3-type with rows,columns,channels, where
  # channels is the number of components in each pixel.  The img.dtype
  # is 'uint8', meaning that each component is an 8-bit unsigned
  # integer.

  img = netpbm.imread( inputFile ).astype('uint8')
  
  # Compress the image
  #
  #
  # Note that single-channel images will have a 'shape' with only two
  # components: the y dimensions and the x dimension.  So you will
  # have to detect this and set the number of channels accordingly.
  # Furthermore, single-channel images must be indexed as img[y,x]
  # instead of img[y,x,1].  You'll need two pieces of similar code:
  # one piece for the single-channel case and one piece for the
  # multi-channel case.

  startTime = time.time()

  outputBytes = bytearray()

  # initialize dictionary
  d = {}
  counter = 256
  for i in range(-counter, counter):
    d[str(i)] = i
  # Set Dictionary limit
  
  # Make a list to hold bytes
  tempBytes = []
  # A counter for the number of bytes
  numBytes = 0
  multichannel = True
  # for a single channel image
  if (len(img.shape) == 2) :
    multichannel = False
    # Go through whole image
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        # Initialize prediction to image value
        prediction = img[y][x]
        #""" 
        # Modify prediction to show the difference between prior pixels and current pixel
        if(x != 0):
          prediction = prediction - img[y][x-1]
        elif(y != 0):
          prediction = prediction - img[y-1][x]
        else:
          prediction = prediction - (img[y][x-1]/3 + img[y-1][x]/3 + img[y-1][x-1]/3)
        #"""
        # Add the predicted value to the bytestream
        tempBytes.append(prediction)
        numBytes += 1

  # Multi-Channel
  else:
    
    # Go through whole image and channels
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        for c in range(img.shape[2]):
          # Do predictive encoding

          # Initialize prediction to image value
          prediction = img[y][x][c]
          # Modify prediction to show the difference between prior pixels and current pixel 
          #""""
          if(x!=0):
            prediction = prediction - img[y][x-1][c]
          elif(y != 0):
            prediction = prediction - img[y-1][x][c]
          else:
            prediction = prediction - (img[y][x-1][c]/3 + img[y-1][x][c]/3 + img[y-1][x-1][c]/3)
          #""""
          # Add the predicted value to the bytestream
          tempBytes.append(prediction)
          numBytes += 1

  # Using a string variable as it allows for concatenation
  s = ""
  # Set s to the first value of the bytestream 
  s = str(tempBytes[0])
  # Go through all bytes
  for i in range(1, numBytes):
    # Do LZW encoding
    # If trying to add entry larger than max size of the dictionary reinitialize the dictionary
    if(counter >= twoBytes):
      counter = 256
      d = {}
      for i in range(-counter, counter):
        d[str(i)] = i

    # Add the next byte to the current string. Uses a delimeter to distinguish numbers
    w = s +"|"+str(tempBytes[i])
    # Checking if it has been seen before
    if w in d:
      s = w
    else:
      # Output bytes by splitting integer into two bytes, this allows for a larger dictionary
      outputBytes.append((int(d[s]) >> 8) & 0xFF)
      outputBytes.append(int(d[s]) & 0xFF)
      # Add to dictionarry
      d[w] = counter
      counter += 1
      s = str(int(tempBytes[i]))
  # Check if the last byte was added or not    
  if s in d:        
    outputBytes.append((int(d[s]) >> 8) & 0xFF)
    outputBytes.append(int(d[s]) & 0xFF)          
  
  endTime = time.time()

  # Output the bytes
  #
  # Include the 'headerText' to identify the type of file.  Include
  # the rows, columns, channels so that the image shape can be
  # reconstructed.

  outputFile.write( '%s\n'       % headerText )
  if (multichannel):
    outputFile.write( '%d %d %d\n' % (img.shape[0], img.shape[1], img.shape[2]) )
    
  else:
    one = 1
    outputFile.write( '%d %d %d\n' % (img.shape[0], img.shape[1]), one )
  outputFile.write( outputBytes )

  # Print information about the compression
  if (multichannel):
    inSize  = img.shape[0] * img.shape[1] * img.shape[2]
  else:
    inSize = img.shape[0] * img.shape[1]
  outSize = len(outputBytes)

  sys.stderr.write( 'Input size:         %d bytes\n' % inSize )
  sys.stderr.write( 'Output size:        %d bytes\n' % outSize )
  sys.stderr.write( 'Compression factor: %.2f\n' % (inSize/float(outSize)) )
  sys.stderr.write( 'Compression time:   %.2f seconds\n' % (endTime - startTime) )
  


# Uncompress an image

def uncompress( inputFile, outputFile ):

  # Check that it's a known file

  if inputFile.readline() != headerText + '\n':
    sys.stderr.write( "Input is not in the '%s' format.\n" % headerText )
    sys.exit(1)
    
  # Read the rows, columns, and channels.  counter

  rows, columns, channels = [ int(x) for x in inputFile.readline().split() ]

  # Read the raw bytes.

  inputBytes = bytearray(inputFile.read())

  # Build the image
  #
  # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes' ARRAY INTO AN IMAGE IN 'img'.
  
  startTime = time.time()

  result = []

  # initialize the dictionary in the opposite was as compress and use an array as the value
  d = {} # create a dictionary
  counter = 256
  
  # Initialize dictionary with values equalling keys from [-256,256]
  for i in range(-counter, counter):
    d[i] = [i]

  img = np.empty( [rows,columns,channels], dtype=np.uint8 )

  byteIter = iter(inputBytes)
  

  # Get encoding in the form of next two bytes
  enc = (byteIter.next() << 8) + byteIter.next()
  s = d[enc]
  #print s
  result.append(s[0])

  for i in range(1, len(inputBytes)//2):
    
    # again reset the dictionary if it reaches the limit
    # Initialize dictionary with values equalling keys from [-256,256]

    if counter >= twoBytes:
      d = {} # initialize blank dictionary
      counter = 256
      for i in range(-counter, counter):
        d[i] = [i]
    
    
    enc = (byteIter.next() << 8) + byteIter.next()

    #retrieve value of dictionary entry from dictionary or create entry assuming it has not yet been entered into dictionary

    if enc in d:
      d_value = d[enc]
    else:
      d_value = []
      for j in s:
        d_value.append(j)
      d_value.append(s[0])
    
    #add dictionary entry value to the result
    for k in range(len(d_value)):
      result.append(d_value[k])

    #Create entry in dictionary
    temp = []
    for j in s:
      temp.append(j)
    temp.append(s[0])
    d[counter] = temp
    counter += 1
   
     
    # reset decoded string to dictionary entry value
    s = d_value


  #implement predictive encoding
  prediction = 0
  counter = 0
  # for a single channel image
  if (channels == 1):
    # Go through whole image
    for y in range(rows):
      for x in range(columns):
        #'''
        if(x != 0):
          prediction = img[y][x-1]
        elif(y != 0):
          prediction = img[y-1][x]
        else:
          prediction = (img[y][x-1]/3 + img[y-1][x]/3 + img[y-1][x-1]/3)
        #'''
        
        img[y,x] = result[counter] + prediction
        counter += 1
  
  for y in range(rows):
    for x in range(columns):
      for c in range(channels):          
        #""""
        if(x != 0):
          prediction = img[y][x-1][c]
        elif(y != 0):
          prediction = img[y-1][x][c]
        else:
          prediction = (img[y][x-1][c]/3 + img[y-1][x][c]/3 + img[y-1][x-1][c]/3)
        #""""
        img[y,x,c] = result[counter] + prediction

        counter += 1

  endTime = time.time()
  # Output the image

  netpbm.imsave( outputFile, img )

  sys.stderr.write( 'Uncompression time: %.2f seconds\n' % (endTime - startTime) )

  

  
# The command line is 
#
#   main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)

# Get input file
 
if sys.argv[2] == '-':
  inputFile = sys.stdin
else:
  try:
    inputFile = open( sys.argv[2], 'r' )
  except:
    sys.stderr.write( "Could not open input file '%s'.\n" % sys.argv[2] )
    sys.exit(1)

# Get output file

if sys.argv[3] == '-':
  outputFile = sys.stdout
else:
  try:
    outputFile = open( sys.argv[3], 'w' )
  except:
    sys.stderr.write( "Could not open output file '%s'.\n" % sys.argv[3] )
    sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
  compress( inputFile, outputFile )
elif sys.argv[1] == 'u':
  uncompress( inputFile, outputFile )
else:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)
