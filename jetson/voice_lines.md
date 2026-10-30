# Hat voice lines — recording / TTS checklist

Audio for the hat's streamed lines. Save each line as `jetson/voice/<id>.wav`
(mp3/m4a also fine — the sender converts to 16-bit mono 22050 Hz automatically).

Per sorting, the Jetson assembles ONE utterance and streams it:

    opening (no name) + NAME clip + [suspense, ~25% of the time] + house ending

with ~250-300 ms pauses between parts (tune `--gap`). So openings never
contain the guest's name — that's what the name clip is for. This list
reuses the name-free (`*_generic`) text from dialogue.yaml, so the mirror's
captions and the hat's voice stay in sync line-for-line.

Test a full line once you have three files:

    python jetson/send_audio.py --host <hat-ip> --gap 300 \
      jetson/voice/dlg_open_01.wav jetson/voice/dlg_name_test.wav \
      jetson/voice/dlg_gryffindor_01.wav

## Progress: 0 / 41 shared lines (+ name clips per guest)

## Openings — 8 lines (id: dlg_open_01..08)
Name-free; each ends leading into the pause where the hat "reads" the guest.

- [ ] dlg_open_01 — "Hmmmm... now this is interesting. Very interesting indeed. Let's see what secrets you're hiding..."
- [ ] dlg_open_02 — "Well, well, well... another mind to examine! Oh, I do hope you're more interesting than the last one."
- [ ] dlg_open_03 — "Ahhh... courage, ambition, wisdom, loyalty... I see a little of everything in here. But which is strongest?"
- [ ] dlg_open_04 — "Oh dear... there's quite a lot happening inside this head. Do try to keep your thoughts organized!"
- [ ] dlg_open_05 — "Another student! Another destiny! And another reason I should have retired a hundred years ago..."
- [ ] dlg_open_06 — "Hmm... an unusual mind. Very unusual. I haven't encountered one quite like yours in years."
- [ ] dlg_open_07 — "Don't be nervous! I've been sorting witches and wizards for centuries. I only get it wrong occasionally."
- [ ] dlg_open_08 — "Ah, yes... I see your strengths, your weaknesses, and... oh my! Let's pretend I didn't see that last thought."

## Name clips — one per guest (id: dlg_name_<first name>)
Just the guest's first name, spoken slowly and thoughtfully — "Alice..." —
as if the hat is turning the name over. Generated when the guest list is
set up (guest -> name clip + house). Examples of the convention:

- [ ] dlg_name_test — a stand-in name for testing the full line
- [ ] dlg_name_<guest> — one per guest on the list

## Suspense — 5 lines, optional (id: dlg_susp_01..05)
Played after the name ~25% of the time, before the verdict.

- [ ] dlg_susp_01 — "Hmmmm... two houses would suit you equally well. How delightfully inconvenient..."
- [ ] dlg_susp_02 — "Wait... wait... what's this? Oh! I certainly wasn't expecting to find THAT inside your head!"
- [ ] dlg_susp_03 — "Ah, I've got it! No, wait. That's not right... or is it? Oh, do be patient! I'm a hat, not a miracle worker!"
- [ ] dlg_susp_04 — "Interesting... very interesting. There is more to you than meets the eye. Much more..."
- [ ] dlg_susp_05 — "The decision has been made... your destiny is calling... and I know exactly where you belong!"

## House endings — 7 per house (id: dlg_<house>_01..07)
Each ends by shouting the house name — that's the reveal.

### Gryffindor
- [ ] dlg_gryffindor_01 — "Ahhh! A heart full of courage! A spirit that refuses to back down! There's only one place for you... GRYFFINDOR!"
- [ ] dlg_gryffindor_02 — "Bold, determined, and always ready to stand up for what's right. The lion's spirit burns brightly within you. GRYFFINDOR!"
- [ ] dlg_gryffindor_03 — "When others run from danger, you stand your ground. Fear may visit you, but it will never control you. GRYFFINDOR!"
- [ ] dlg_gryffindor_04 — "Oh, you're brave! Very brave! Unfortunately, your common sense seems to have taken the day off. Definitely GRYFFINDOR!"
- [ ] dlg_gryffindor_05 — "You'd charge into a dragon's cave without a plan, wouldn't you? Good heavens... another one! GRYFFINDOR!"
- [ ] dlg_gryffindor_06 — "I see danger in your future... impossible challenges... and a courage that will shake the very walls of Hogwarts! GRYFFINDOR!"
- [ ] dlg_gryffindor_07 — "There's a fire inside you that refuses to be extinguished. Whatever obstacles you face, you'll find the strength to overcome them. GRYFFINDOR!"

### Slytherin
- [ ] dlg_slytherin_01 — "Ahhh... ambition! Determination! A mind always three steps ahead. Oh, I know exactly where you belong. SLYTHERIN!"
- [ ] dlg_slytherin_02 — "You understand that greatness is not simply given. It must be earned, pursued, and sometimes taken. SLYTHERIN!"
- [ ] dlg_slytherin_03 — "A sharp mind, a stronger will, and the instincts of a serpent. Your path to greatness begins in SLYTHERIN!"
- [ ] dlg_slytherin_04 — "Oh, you've already planned how to take over Hogwarts, haven't you? How wonderfully concerning. SLYTHERIN!"
- [ ] dlg_slytherin_05 — "Clever, charming, and just a little suspicious. Remind me never to play cards against you. SLYTHERIN!"
- [ ] dlg_slytherin_06 — "Power does not frighten you. Greatness does not escape you. And destiny? You intend to command it! SLYTHERIN!"
- [ ] dlg_slytherin_07 — "You see possibilities where others see obstacles. You know what you want, and you possess the determination to achieve it. SLYTHERIN!"

### Ravenclaw
- [ ] dlg_ravenclaw_01 — "Ohhh... what a fascinating mind! So many questions, so many possibilities! A brilliant future awaits you in RAVENCLAW!"
- [ ] dlg_ravenclaw_02 — "You seek knowledge not simply for its power, but for the wonder of discovery. A true scholar at heart. RAVENCLAW!"
- [ ] dlg_ravenclaw_03 — "An inventive mind, a restless imagination, and a love for mysteries. Your place is surely in RAVENCLAW!"
- [ ] dlg_ravenclaw_04 — "You question absolutely everything! Including my methods, apparently. How delightfully irritating. RAVENCLAW!"
- [ ] dlg_ravenclaw_05 — "You'd stop in the middle of a dragon attack just to ask how dragon fire works. Fascinating... and completely ridiculous. RAVENCLAW!"
- [ ] dlg_ravenclaw_06 — "I see visions of discovery... secrets waiting to be uncovered... and ideas that may change the magical world forever! RAVENCLAW!"
- [ ] dlg_ravenclaw_07 — "Your greatest gift is your ability to see the world differently. Never stop questioning, imagining, and discovering. RAVENCLAW!"

### Hufflepuff
- [ ] dlg_hufflepuff_01 — "Ahhh... a kind heart, a loyal spirit, and a soul as warm as sunshine. There can be no doubt. HUFFLEPUFF!"
- [ ] dlg_hufflepuff_02 — "You value friendship over glory, fairness over power, and kindness above all else. A true HUFFLEPUFF!"
- [ ] dlg_hufflepuff_03 — "Patience, determination, and unwavering loyalty. Never underestimate the strength of a gentle heart. HUFFLEPUFF!"
- [ ] dlg_hufflepuff_04 — "Oh, what a lovely mind! You're even thinking about sharing your snacks with everyone. HUFFLEPUFF!"
- [ ] dlg_hufflepuff_05 — "You'd apologize to a spider before carrying it outside, wouldn't you? Oh, you're definitely a HUFFLEPUFF!"
- [ ] dlg_hufflepuff_06 — "In a world filled with darkness, some souls shine with extraordinary warmth. Your kindness will become your greatest magic! HUFFLEPUFF!"
- [ ] dlg_hufflepuff_07 — "You may not seek the spotlight, but the people around you know your worth. Your loyalty and compassion make you extraordinary. HUFFLEPUFF!"

## Extras — optional, hat-only, add later if wanted
Not part of a sorting; played between guests or when the camera sees no
known face.

- [ ] dlg_idle_01 — quiet muttering ("Nobody sitting down? Fine... I'll just sit here... sorting... the air...")
- [ ] dlg_idle_02 — "(Nobody ever asks how the hat is doing...)"
- [ ] dlg_unknown_01 — "Hmm! A new face. Sit down, sit down — and do tell me your name..."
