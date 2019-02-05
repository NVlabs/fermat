#!/usr/bin/env python

import argparse
import re
import os

def main():
	parser = argparse.ArgumentParser(description='Process cross-references in code blocks.')
	parser.add_argument('source', metavar='SOURCE', type=str, help='source file')
	parser.add_argument('dest', metavar='DEST', type=str, nargs='?', help='output file')
	args = parser.parse_args()

	with open(args.source, mode='r') as file_in:
		content = file_in.read()

	dest = args.dest
	if (args.dest is None) or (args.dest == args.source):
		dest = args.source
		os.rename(args.source, "{}.orig".format(args.source))

	n_links = 0
	output = ''
	ref_re = re.compile(r'(.*?)(\s*)&lt;&lt;\s+([a-zA-Z0-9_ -]+)\s+&gt;&gt;(.*)', re.IGNORECASE | re.DOTALL | re.UNICODE)
	rest = content
	while len(rest) > 0:
		m = ref_re.match(rest)
		if not m:
			break
		(pre, spaces, title, rest) = m.groups()
		output += pre
		output += spaces
		anchor = title.replace(" ", "_") + "_anchor"
		# print("{} -> {}".format(title, anchor))
		link = '<a href="#{}">&lt;&lt; {} &gt;&gt;</a>'.format(anchor, title)
		output += link
		n_links += 1
	output += rest

	with open(dest, mode='w') as file_out:
		file_out.write(output)

	print("Output written to {!r}. {} links processed.".format(dest, n_links))


if __name__ == '__main__':
	main()
