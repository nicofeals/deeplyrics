import lyricsgenius
import os

genius = lyricsgenius.Genius(os.environ['GENIUS_ACCESS_TOKEN'])
artists = [
    'Casey (UK)',
    'Holding Abscence',
    'nothing,nowhere.',
    'Being as an Ocean',
    'Touché Amoré',
    'Counterparts',
    'Hundredth',
    'Worthwile',
    'Landmvrks',
    'Ambleside',
    'Endless Heights',
    'Rarity',
    'Novelists FR',
    'Alazka',
    'Architects',
    'Bring Me The Horizon',
    'Sleep Token',
    'Bullet For My Valentine',
    'Modern Error',
    'Invent Animate',
    'Northlane',
    'Parkway Drive',
    'The Ghost Inside',
    'As I lay Dying',
    'Beartooth',
    'Wage War',
    'While She Sleeps',
    'Polaris',
    'Polar',
    'Bury Tomorrow',
    'In Hearts Wake',
    'Bad Omens',
    'Fit for a king',
    'Spiritbox',
    'Our Hollow, Our Home',
    'Dayseeker',
    'Erra',
    'Breakdown of sanity',
    'It Prevails',
    'Stick to your guns',
    'Deez Nuts',
    'Napoleon',
    'Hollow Front',
]


for name in artists:
    artist = genius.search_artist(name, max_songs=20, sort='popularity')
    print(artist.songs)
    artist.save_lyrics()