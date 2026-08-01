# Third-party notices

Hashdrills embeds the following browser assets so rich drill content works
without a CDN.

## KaTeX 0.17.0

Source: <https://github.com/KaTeX/KaTeX/tree/v0.17.0>

`vendor/katex/katex.min.js`, `vendor/katex/katex.min.css`, and the KaTeX
documentation are from the KaTeX 0.17.0 distribution. Hashdrills changes only
the font URLs in `katex.min.css`, from KaTeX's relative `fonts/` directory to
Hashdrills' `/assets/katex/fonts/` route. The JavaScript is unmodified.

Copyright (c) 2013-2020 Khan Academy and other contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### KaTeX mhchem extension

Source: <https://github.com/KaTeX/KaTeX/blob/v0.17.0/contrib/mhchem/mhchem.js>

`vendor/katex/contrib/mhchem.min.js` is the unmodified KaTeX 0.17.0 build of
the mhchem extension. Its source describes the KaTeX implementation as an
adaptation of `MathJax/extensions/TeX/mhchem.js` for mhchem 3.3.0. The KaTeX
source marks its adaptation as MIT-licensed while retaining the adapted
MathJax work's Apache License 2.0 notice:

Copyright (c) 2011-2015 The MathJax Consortium  
Copyright (c) 2015-2018 Martin Hensel

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at:

<https://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License. The full Apache License 2.0 text is also included in this
distribution's `LICENSE` file.

### KaTeX fonts

Upstream font source and license metadata:
<https://github.com/KaTeX/KaTeX/blob/v0.17.0/src/fonts/lib/Space.ttx>

The unmodified KaTeX 0.17.0 WOFF2 files under `vendor/katex/fonts/` cover these
font families: `KaTeX_AMS`, `KaTeX_Caligraphic`, `KaTeX_Fraktur`,
`KaTeX_Main`, `KaTeX_Math`, `KaTeX_SansSerif`, `KaTeX_Script`, `KaTeX_Size1`,
`KaTeX_Size2`, `KaTeX_Size3`, `KaTeX_Size4`, and `KaTeX_Typewriter`.

Copyright (c) 2009-2010 Design Science, Inc. (<https://www.mathjax.org>)  
Copyright (c) 2014-2018 Khan Academy (<https://www.khanacademy.org>)

The font families are licensed under the SIL Open Font License, Version 1.1,
and their family names are Reserved Font Names. The copyright header, Reserved
Font Names, and complete license are included in
`vendor/katex/fonts/OFL.txt`. KaTeX's upstream build notes that its font
generation was originally based on MathJax font generation.

## highlight.js 11.9.0

Source: <https://github.com/highlightjs/cdn-release/tree/11.9.0/build>

`vendor/highlight/highlight.js` and the GitHub theme in
`vendor/highlight/highlight.css` are unmodified files from the official
highlight.js 11.9.0 CDN release.

Copyright (c) 2006, Ivan Sagalaev. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
