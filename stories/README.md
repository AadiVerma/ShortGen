# Stories to post

Short scripts written to be rendered with the default local pipeline and posted as YouTube Shorts. They are a test: nine scripts in different formats, to see which one real viewers respond to. Each folder holds a `script.md` (the narration, with a picture description under every scene) and a `config.yaml` (voice, pacing, picture style, caption style).

They are separate from `examples/`, which exists to demonstrate formats and to be run by the end-to-end test.

## The stories

| Folder | Format | Scenes | Length | What it is |
|---|---|---|---|---|
| `one-light-day` | Space fact with a date | 8 | 47 s | On 18 November 2026 Voyager 1 is one light-day from Earth. **Time-sensitive: post it on or before that day** |
| `vanished-keepers` | True unsolved mystery | 8 | 44 s | The three lighthouse keepers who disappeared from the Flannan Isles in December 1900 |
| `immortal-jellyfish` | One animal fact | 9 | 29 s | The jellyfish that turns back into its younger form instead of dying |
| `golden-repair` | Short reflection | 7 | 27 s | Kintsugi, and what it says about your own cracks |
| `count-the-lights` | Original horror | 8 | 39 s | A lake house, a rule, and one lantern too many. Fiction written for this project |
| `town-on-fire` | Strange true history | 8 | 40 s | Centralia, Pennsylvania, burning underground since 1962 |
| `pip-the-owl` | Story for young children | 10 | 49 s | A little owl who cannot hoot, and the foggy night her whistle helps. Original, for ages 4 to 8, written to be joined in with. **Read [For children](#for-children) before posting** |
| `moon-night-off` | Bedtime story for young children | 10 | 48 s | The moon takes a night off, everyone fails to replace her, and a firefly takes a smaller job. Original, for ages 4 to 8. Written after `pip-the-owl` was judged lame: it starts from an idea, not a lesson. **Read [For children](#for-children) before posting** |
| `thirsty-crow` | Traditional fable for young children | 9 | 48 s | Aesop's crow, the tall pot and the pebbles, retold in new words for ages 3 to 7. **Read [For children](#for-children) before posting** |

Lengths are from rendering each one with the real voice; see "What was checked" below.

## Making one

```bash
shortgen run stories/one-light-day
```

The video is written to `stories/one-light-day/output/final.mp4`, with `subtitles/captions.srt` beside it.

On the 16 GB Mac this project was measured on, the default image model took about 4.5 minutes per picture at 30 steps. These stories ask for 24 steps, which took about 3 minutes per picture there (two pictures timed), so allow about 25 minutes per story. Close other apps first and keep the laptop on a hard surface; it will get hot.

Look at the pictures in `images/` before you post. The model does not always draw what it was asked for. To change one picture, edit the `image:` line under that scene in `script.md` and run the same command again: only that scene is redrawn. Running again without changing anything redraws nothing, and the same description always gives the same picture.

## What these scripts were written around

The first real run of the image model showed what it is good and bad at, and the scripts avoid the bad parts:

- **One subject per picture, described by what can be seen.** The model draws "a lighthouse in a storm" well and ignores "three hearts inside its body".
- **No people's faces, no hands, no recurring human character.** The model draws each scene from scratch, so a person would look different every time. Places, objects, animals and weather hold together across scenes.
- **No text, diagrams, maps or exact counts in pictures.** The horror story never shows its seven lanterns as seven.
- **The style line says how to paint, not what genre it is.** The first style tried for `one-light-day` began "retro space illustration"; the model answered with abstract art-deco shapes and a man standing on an alien planet. A genre name brings its own subjects. "No people" in a style is also read as "people"; exclusions go in `negative_prompt`.
- **Keep things out of one picture with a `negative:` line.** The project's `negative_prompt` applies to every picture, so it cannot say "no moon" in a story that also shows the moon. A `<!-- negative: ... -->` line under a scene replaces it for that picture alone. `moon-night-off` uses it to keep lamps out of a dark bedroom.
- **Adding a scene redraws every picture after it.** A scene's picture depends on its position in the script. To add a beat without losing good pictures, change what an existing scene shows instead.
- **Motion is a slow zoom or pan over a still.** Scripts cut to a new picture every four or five seconds so the video does not feel static.

## Posting them

Suggested titles, descriptions and a first comment. Change them to sound like you.

| Folder | Title | Description | First comment to pin |
|---|---|---|---|
| `one-light-day` | On Nov 18, Voyager 1 will be one light-day from Earth | Launched in 1977, Voyager 1 reaches one light-day from Earth (25.9 billion km) on 18 November 2026, according to NASA. A radio signal now takes 24 hours to get there. #voyager #space #nasa | If you could add one sound to the golden record, what would it be? |
| `vanished-keepers` | Three lighthouse keepers vanished in 1900. They were never found. | The Flannan Isles lighthouse, Scotland, December 1900. Every detail here is from the official records held by the National Records of Scotland. #mystery #history #unsolved | A wave, or something else? What do you think happened? |
| `immortal-jellyfish` | This animal can age backwards | Turritopsis dohrnii, the "immortal jellyfish", can turn back into its younger form when it is hurt or starving. It is about 4.5 mm across. #animals #ocean #science | Would you start over if you could? |
| `golden-repair` | Why broken bowls are mended with gold | Kintsugi means "golden joinery": mending broken pottery with lacquer and powdered gold, so the repair shows. #kintsugi #wisdom #mindset | What is one crack you stopped hiding? |
| `count-the-lights` | Do not count the lights on the water | An original short horror story. #horror #scarystories #shortstory | Would you have counted? |
| `town-on-fire` | This town has been on fire since 1962 | Centralia, Pennsylvania. A coal seam under the town has been burning since May 1962 and could burn for another century. #history #centralia #weirdhistory | Would you have stayed? |
| `pip-the-owl` | The Little Owl Who Could Not Hoot | A short bedtime story about a little owl whose whistle helps a lost rabbit find the way home. An original story about being different. #bedtimestory #kidsstories #owl | None: comments are off on videos made for kids |
| `moon-night-off` | The Night the Moon Took the Night Off | The moon is tired, so she takes one night off. Who will light up the dark? A short bedtime story for little ones who do not like the dark. #bedtimestory #kidsstories #moon | None: comments are off on videos made for kids |
| `thirsty-crow` | The Thirsty Crow | The clever crow, the tall pot and the pebbles: a classic story for little ones about not giving up. #thirstycrow #moralstories #kidsstories | None: comments are off on videos made for kids |

To learn something from the test, change one thing at a time: post one a day at about the same hour, do not promote any of them, and after 48 hours compare the same numbers for each in YouTube Studio: views, how many viewers chose to watch rather than swipe away, the average share of the video watched, and comments. Six videos is a small sample, so treat the result as a hint about which format to make more of, not as proof.

## YouTube's rules, as checked on 11 November 2026

These were read from YouTube's help pages and one trade article on that date. Rules change; check before you rely on them. This is not legal advice.

- **They qualify as Shorts.** A Short is a square or vertical video up to three minutes long. These are vertical and under a minute.
- **AI disclosure.** YouTube Studio has an "AI use" setting on every upload. Disclosure is required for realistic altered or synthetic content, and is not required for animation or clearly unrealistic content, which is what these illustrated styles are. YouTube says disclosing does not limit a video's audience or its ability to earn money, and it may add a label on its own if its systems detect AI content. If a picture comes out looking like a real photograph of a real place (possible in `vanished-keepers` and `town-on-fire`), or you are unsure, choose "Yes".
- **Do not expect these to earn money as they are.** YouTube does not pay for what it calls inauthentic content, and its July 2026 clarification names image slideshows, templated storylines and AI content made from generic templates without the creator's own perspective. Uploading such videos is allowed; monetising a channel made only of them is unlikely. Treat this round as a test of what people watch. A channel worth monetising needs your own voice, opinions or footage on top.
- **Rights.** The scripts are original to this project and under its Apache-2.0 license. The default voice model (Kokoro) is Apache-2.0. The default image model (Stable Diffusion XL) permits commercial use of its output, with restrictions on harmful uses; see `docs/licenses.md`. There is no music in these videos.

## For children

`pip-the-owl`, `moon-night-off` and `thirsty-crow` are meant for young children, and different rules apply to them.

**Why this kind of story.** The best data found on what children watch is a survey of 1,129 Australian children aged 6 to 13 (Roy Morgan, April to December 2025): gaming 53%, animation 37%, comedy 36%, animals 34%, music 34%; among 6 to 9 year olds, animation 40% and animals 35%. It covers one country and YouTube as a whole, not Shorts; nothing specific to children and Shorts was found. An animated animal story is where the top categories overlap with what this pipeline can make. Gaming, challenges and music are out of its reach.

**What the most-watched children's videos have in common.** In a June 2026 snapshot, five of YouTube's ten most-viewed videos of all time were for small children, and all five are songs ("Baby Shark Dance" at about 17 billion views, then "Wheels on the Bus", "Bath Song", "Johny Johny Yes Papa" and a phonics song at 7 to 9 billion each). The analysis credits repetition, simple words, things to copy, and the fact that a child asks for the same video again and again. Advice for children's Shorts adds a question or teaser at the start, and a puzzle with a moment to think before the answer. This pipeline cannot make songs, which is the largest thing it lacks for this audience. It can use the rest, and both stories do: a question or problem in the first second, sounds to copy, words in threes, a pause to think, a short lesson.

**Similar, not identical.** Those videos' scripts, characters, songs and look belong to their makers, and copying them is both wrong and a fast way to a takedown. What can be used freely is their shape, and stories old enough to belong to everyone. `thirsty-crow` is Aesop's "The Crow and the Pitcher", one of the most retold children's stories on YouTube, told here in new words with new pictures. `pip-the-owl` and `moon-night-off` are original.

**What the first one taught.** `pip-the-owl` was built from a checklist: a hero who is different, a problem, a lesson said out loud. It was posted, and the honest verdict on it was that it is lame, which is fair: it has a moral and no idea, and nothing in it is funny or surprising. `moon-night-off` was written the other way round, from one idea (the moon is tired), with jokes a child gets, no stated moral, and a last line that points at the child's own room. Its hero is a firefly, a dot of light, so he cannot look different from picture to picture the way the owl did.

**How they were written.** YouTube publishes what it considers good children's content: kindness and looking after others, feelings a child can recognise, original stories, and a complete arc with a clear lesson. It lists what it considers poor: hard to follow, sensational or misleading, deceptively educational, encouraging bad behaviour, heavily promotional. Both stories follow the first list: one small hero, one problem, a kind ending, a lesson said once. Nothing in them is frightening; the image settings exclude scary pictures.

**How `pip-the-owl` tries to hold a small child.** It opens by asking the child a question ("Can you hoot like an owl?"), gives sounds to copy and repeats them (Hoo! Twee! Hop!), and ends by asking the child to whistle. The captions show each sound as one big word. The first picture is a close-up of a friendly face looking straight out, and the pictures are bright and rounded. None of this has been tested on children; it follows how picture books and children's shows are built. The voice is the model's best-rated one; whether a child likes it is for a listener to judge, and `speech.voice` in the story's `config.yaml` changes it.

**What you must do when uploading.**

- **Set the audience to "Yes, it's made for kids."** This is a legal requirement for content directed at children, not a preference. YouTube may act on the account if it is set wrongly.
- **Expect features to be off.** On made-for-kids videos YouTube turns off comments, the notification bell, personalised ads, saving to playlists and several others. You will not get comments to read, so judge this one by views and watch time only.
- **Watch every picture yourself before posting.** The image model has not drawn these stories yet, and a strange or unsettling picture matters much more for small children than for adults.

**What to expect.** YouTube says it limits AI-made content in the YouTube Kids app to a small group of vetted channels, so these will be seen on ordinary YouTube, not there. Its guidance also says a channel focused on low-quality children's content can lose monetisation, and names mass production and auto-generation as a common cause. In 2026 more than 200 child specialists and groups asked YouTube to stop recommending AI-made videos to children. A few careful, original stories are defensible. A stream of generated ones is exactly what the criticism is about.

## What was checked, and what was not

Checked: every script, the three children's stories included, was rendered end to end with the real voice model and the real renderer, with labelled placeholder pictures standing in for the image model. That confirms the narration, the timings, the captions and the final video for each one. The facts in the four non-fiction scripts were checked against the sources below.

Not checked: the pictures, for seven of the nine. `one-light-day` and `pip-the-owl` have been drawn by the image model and rendered in full, and each took three rounds of rewording; the owl still does not look the same in every picture. The other seven have not been drawn at all. Expect the same, and try two scenes before committing to a whole story:

```bash
shortgen generate stories/moon-night-off --stage image --scene s001 --scene s004
```

## Sources

- Voyager 1: [EarthSky on Voyager 1 reaching one light-day from Earth](https://earthsky.org/space/voyager-1-1-light-day-from-earth-november-17-18-2026/)
- Flannan Isles: [National Records of Scotland, "The Flannan Islands lighthouse: a disappearance"](https://nrscotland.gov.uk/learning-and-events/for-the-safety-of-all-scotlands-lighthouses/the-flannan-islands-lighthouse-a-disappearance/)
- Immortal jellyfish: [American Museum of Natural History, "The Immortal Jellyfish"](https://www.amnh.org/explore/news-blogs/on-exhibit-posts/the-immortal-jellyfish)
- Centralia: [History.com on the Centralia mine fire](https://www.history.com/news/mine-fire-burning-more-50-years-ghost-town)
- Shorts length: [YouTube Help, "Understand three-minute YouTube Shorts"](https://support.google.com/youtube/answer/15424877?hl=en)
- AI disclosure: [YouTube Help, "Disclosing use of GenAI content"](https://support.google.com/youtube/answer/14328491?hl=en-IN&co=GENIE.Platform%3DDesktop)
- What children watch: [Mediaweek, February 2026, on Roy Morgan's Young Australian Survey](https://www.mediaweek.com.au/analysis/youtube-tightens-grip-on-australian-kids-as-gaming-and-mrbeast-dominate)
- Most-viewed children's videos: [Webiano, June 2026, on YouTube's all-time top ten](https://webiano.digital/youtubes-all-time-top-ten-is-a-childrens-media-story-before-it-is-a-music-story/)
- Children's Shorts formats: [AIR Media-Tech, Shorts ideas for kids channels](https://air.io/en/youtube-hacks/youtube-shorts-ideas-for-youtube-kids-channels)
- Children's content quality: [YouTube Help, "Best practices for kids & family content"](https://support.google.com/youtube/answer/10774223?hl=en)
- Made for kids: [YouTube Help, "Set your channel or video's audience"](https://support.google.com/youtube/answer/9527654?hl=en&co=GENIE.Platform%3DDesktop)
- AI videos for children: [Android Authority on the letter to Google and YouTube's response](https://www.androidauthority.com/google-warned-ai-videos-for-kids-on-youtube-3654082)
- Monetisation: [Tubefilter, 13 July 2026, on YouTube's inauthentic content policy](https://www.tubefilter.com/2026/07/13/youtube-inauthentic-content-monetization-policy-update/)
