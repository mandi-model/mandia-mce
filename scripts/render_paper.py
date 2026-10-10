"""Render the observed French article to a readable, editable-source-backed PDF."""
from pathlib import Path
import sys,re,html,os
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

source=Path(sys.argv[1]);dest=Path(sys.argv[2]);base=source.parent
fonts=Path(os.environ.get('MCE_FONTS_DIR','/usr/share/fonts/truetype/dejavu'))
if not fonts.is_dir(): raise ValueError('Set MCE_FONTS_DIR to the folder containing DejaVuSans TrueType fonts')
for name,file in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf'),('DVI','DejaVuSans-Oblique.ttf'),('DVBI','DejaVuSans-BoldOblique.ttf'),('DVM','DejaVuSansMono.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DVI',boldItalic='DVBI')
WIDTH=A4[0]-92
styles={
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.4,leading=14.1,spaceAfter=8,textColor=colors.HexColor('#202b34')),
 'title':ParagraphStyle('title',fontName='DVB',fontSize=18,leading=24,spaceAfter=15,textColor=colors.HexColor('#155d68')),
 'h2':ParagraphStyle('h2',fontName='DVB',fontSize=12.3,leading=17,spaceBefore=14,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#155d68')),
 'h3':ParagraphStyle('h3',fontName='DVB',fontSize=10.5,leading=15,spaceBefore=10,spaceAfter=6,keepWithNext=True,textColor=colors.HexColor('#354d59')),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=7.2,leading=10,textColor=colors.HexColor('#202b34')),
 'caption':ParagraphStyle('caption',fontName='DVI',fontSize=8.2,leading=12,spaceAfter=10,textColor=colors.HexColor('#5a6972')),
}
def markup(s):
 s=s.replace('—','-').replace('–','-').replace('\u2011','-')
 s=html.escape(s,quote=False)
 s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:'<link href="'+html.escape(m.group(2),quote=True)+'" color="#126d78">'+m.group(1)+'</link>',s)
 s=re.sub(r'`([^`]+)`',lambda m:'<font name="DVM" size="8">'+m.group(1)+'</font>',s)
 s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
 return s
story=[];lines=source.read_text().splitlines();i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
   i+=1
  cols=len(rows[0]);assert all(len(r)==cols for r in rows)
  if cols==6:weights=[1.02,.7,1.02,1.02,1.02,1.55]
  elif cols==5:weights=[1.08,1.1,1.1,1.1,1.1]
  elif cols==4:weights=[1.0,1.3,1.3,1.3]
  else:weights=[1.4]+[1.]*(cols-1)
  if len(weights)!=cols:weights=[1.]*cols
  widths=[WIDTH*w/sum(weights) for w in weights]
  values=[[Paragraph(('<b>'+markup(c)+'</b>') if ri==0 else markup(c),styles['cell']) for c in row] for ri,row in enumerate(rows)]
  t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0f1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#839da2')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d6dfe2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fa')])]))
  if len(rows)<=6:story.append(KeepTogether([t,Spacer(1,10)]))
  else:story.extend([t,Spacer(1,10)])
  continue
 img=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',line)
 if img:
  path=base/img.group(2);assert path.is_file()
  w,h=ImageReader(str(path)).getSize();height=WIDTH*h/w
  flow=Image(str(path),width=WIDTH,height=height)
  flow.keepWithNext=True;flow.spaceAfter=5
  story.extend([flow,Paragraph(markup(img.group(1)),styles['caption'])]);i+=1;continue
 if line.startswith('# '):story.append(Paragraph(markup(line[2:]),styles['title']));i+=1;continue
 if line.startswith('### '):story.append(Paragraph(markup(line[4:]),styles['h3']));i+=1;continue
 if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['h2']));i+=1;continue
 paragraph=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','![')):
  paragraph.append(lines[i].strip());i+=1
 story.append(Paragraph(markup(' '.join(paragraph)),styles['body']))
for start,flow in enumerate(story):
 if isinstance(flow,Paragraph) and flow.text in ('Références','References'):
  stop=next((j for j in range(start+1,len(story)) if isinstance(story[j],Paragraph) and story[j].style.name=='h2'),len(story))
  story[start:stop]=[KeepTogether(story[start:stop])]
  break
def page(canvas,doc):
 canvas.saveState();canvas.setFont('DV',7.4);canvas.setFillColor(colors.HexColor('#667780'))
 canvas.drawString(46,A4[1]-28,'MANDIA RESEARCH  /  EXPLORATORY STUDY' if source.name.endswith('.en.md') else 'MANDIA RESEARCH  /  ÉTUDE EXPLORATOIRE')
 canvas.setStrokeColor(colors.HexColor('#d2dee1'));canvas.line(46,A4[1]-34,A4[0]-46,A4[1]-34)
 canvas.drawString(46,25,'Version 1.0 - 10 October 2026' if source.name.endswith('.en.md') else 'Version 1.0 - 10 octobre 2026')
 canvas.drawRightString(A4[0]-46,25,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=48,bottomMargin=43,title=source.read_text().splitlines()[0].removeprefix('# '),author='mandia',subject='Exploratory experimental study on a French legal glossary' if source.name.endswith('.en.md') else 'Étude expérimentale exploratoire sur un glossaire français',pageCompression=1)
doc.build(story,onFirstPage=page,onLaterPages=page)
print(dest,dest.stat().st_size)
