#!/usr/bin/env node
// Scrape Bing Images for images.

var fs = require('fs');
var request = require('request').defaults({encoding: null});
var argparse = require('argparse');
var crypto = require('crypto');
var sanitize = require('sanitize-filename');
var Scraper = require('images-scraper');

function ensure_dir(path) {
  // Eh... good enough.
  if (!fs.existsSync(path)) {
    fs.mkdirSync(path);
  }
}

// Parse arguments.
var parser = new argparse.ArgumentParser({
  version: '0.0.1',
  addHelp: true,
  description: 'x-or-y image scraper'
});
parser.addArgument(
  [ '-n', '--number' ],
  {
    help: 'Number of results to download for the search query.',
    type: 'int',
    defaultValue: 100
  }
);
parser.addArgument(
  [ 'search-terms' ],
  {
    help: 'Search terms to use.'
  }
);
parser.addArgument(
  [ '-o', '--output-directory' ],
  {
    help: 'Output directory for downloaded images. Defaults to images/<search terms>',
 }
);
var args = parser.parseArgs();
if (args['output_directory'] === null) {
  // Make sure the images directory exists.
  ensure_dir('images');
  args['output_directory'] = 'images/' + sanitize(args['search-terms']);
}
console.log('Getting top %d results for "%s", saving to: %s', args['number'], args['search-terms'], args['output_directory']);

// Make sure our output directory exists.
ensure_dir(args['output_directory']);

// Do the actual scraping.
var bing = new Scraper.Bing();
bing.list({
  keyword: args['search-terms'],
  num: args['number']
}).then(function(results) {
  function get_rest(results) {
    if (results.length == 0)
	  return;
	var entry = results[0];
    console.log('Downloading: %s', entry['thumb']);
    request.get(entry['thumb'], function(err, res, body) {
      hash = crypto.createHash('sha256').update(body).digest('hex');
      console.log('Got: %s', hash);
	  fs.writeFileSync(args['output_directory'] + '/' + hash + '.jpg', body);
      setTimeout(function() { get_rest(results.slice(1)); }, 200);
    });
  }
  get_rest(results);
});

