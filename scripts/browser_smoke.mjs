// Optional verification: installed Chrome + Node 22+, zero npm packages.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {get} from 'node:http';
import {mkdirSync,mkdtempSync,existsSync,readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
const root=resolve(import.meta.dirname,'..');
const cache=resolve(root,'.cache/browser');mkdirSync(cache,{recursive:true});
const profile=mkdtempSync(resolve(cache,'profile-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
 '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--disable-background-networking','--disable-component-update','--disable-sync',
 '--disable-extensions','--disable-breakpad','--disable-crash-reporter',
 '--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
let socket;let id=0;const pending=new Map(),errors=[],remote=[];
const httpJson=url=>new Promise((resolve,reject)=>get(url,r=>{let data='';r.on('data',c=>data+=c);r.on('end',()=>{try{resolve(JSON.parse(data))}catch(e){reject(e)}})}).on('error',reject));
function command(method,params={}){return new Promise((resolve,reject)=>{const key=++id;const timer=setTimeout(()=>{pending.delete(key);reject(new Error('Timeout: '+method))},10000);pending.set(key,{resolve:x=>{clearTimeout(timer);resolve(x)},reject:e=>{clearTimeout(timer);reject(e)}});socket.send(JSON.stringify({id:key,method,params}))})}
async function evaluate(expression){const r=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value}
async function waitFor(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(50)}throw new Error('Waiting for '+expression)}
async function click(selector){await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`)}
try{
 const portFile=resolve(profile,'DevToolsActivePort');
 for(let i=0;i<100&&!existsSync(portFile);i++)await delay(100);
 assert.ok(existsSync(portFile),'Chrome did not start');
 const port=readFileSync(portFile,'utf8').split('\n')[0];
 const targets=await httpJson(`http://127.0.0.1:${port}/json/list`);
 socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});
 socket.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(m.error.message)):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Network.requestWillBeSent'){const u=m.params.request.url;if(/^https?:/.test(u)&&!u.startsWith('http://127.0.0.1:8765/'))remote.push(u)}};
 await command('Runtime.enable');await command('Network.enable');await command('Page.enable');
 // Deny every non-loopback HTTP request even if future edits accidentally add one.
 await command('Network.setBlockedURLs',{urls:['https://*']});
 await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await command('Page.navigate',{url:'http://127.0.0.1:8765/'});
 await waitFor(`document.getElementById('shapeA')?.textContent.includes('int64')`);
 assert.equal(await evaluate(`document.querySelectorAll('.drill-card').length`),5);
 await click('[data-op="transpose"]');await waitFor(`document.getElementById('result-op').textContent==='transpose'`);
 assert.match(await evaluate(`document.getElementById('result').textContent`),/1\s+4\s+7/);
 await click('[data-op="matmul"]');await waitFor(`document.getElementById('result-op').textContent==='matmul'`);
 assert.match(await evaluate(`document.getElementById('result').textContent`),/30/);
 await click('[data-op="softmax"]');await waitFor(`document.getElementById('result-op').textContent==='softmax'`);
 assert.equal(await evaluate(`document.getElementById('axis').value`),'-1');
 await click('#useA');await waitFor(`document.getElementById('arrayA').value.includes('0.')`);
 // Debounced edits, malformed B does not block one-array operations.
 await evaluate(`document.getElementById('arrayB').value='oops';document.getElementById('arrayB').dispatchEvent(new Event('input'))`);
 await click('[data-op="mean"]');await waitFor(`document.getElementById('result-op').textContent==='mean'`);
 await click('[data-op="matmul"]');await waitFor(`document.getElementById('terminal').classList.contains('error')`);
 for(const op of ['standardize','softmax','nearest_center','least_squares','convolution']){
  await click('#tab-drills');await click(`[data-reveal="${op}"]`);
  assert.equal(await evaluate(`document.getElementById('ref-${op}').hidden`),false);
  await click(`[data-drill="${op}"]`);await waitFor(`document.getElementById('result-op').textContent==='${op}'`);
  assert.equal(await evaluate(`document.getElementById('calculator').hidden`),false);
  assert.ok(await evaluate(`document.getElementById('steps').children.length`));
 }
 for(const example of ['linear_gd','logistic','knn','gini','stump','cross_entropy','tensor_puzzle']){
  await click(`[data-example="${example}"]`);await waitFor(`document.getElementById('result-op').textContent==='${example}'`);
 }
 await click('[data-example="gini"]');await waitFor(`document.getElementById('result-op').textContent==='gini'`);
 await click('#useA');await waitFor(`document.getElementById('shapeA').textContent.includes('shape ()')`);
 await click('[data-op="expand_dims"]');await waitFor(`document.getElementById('result-op').textContent==='expand_dims'`);
 assert.match(await evaluate(`document.getElementById('result-meta').textContent`),/\(1,\)/);
 await click('[data-preset="3×3"][data-array="A"]');await click('[data-preset="3×3"][data-array="B"]');
 await click('[data-op="matmul"]');await waitFor(`document.getElementById('result-op').textContent==='matmul'`);
 await evaluate(`window.scrollTo(0,0)`);await delay(100);
 const desktop=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});writeFileSync(resolve(cache,'desktop.png'),Buffer.from(desktop.data,'base64'));
 await click('#tab-drills');await evaluate(`window.scrollTo(0,0)`);
 const drillShot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});writeFileSync(resolve(cache,'drills.png'),Buffer.from(drillShot.data,'base64'));
 await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await click('#tab-calculator');await evaluate(`window.scrollTo(0,0)`);await delay(100);
 assert.ok(await evaluate(`document.documentElement.scrollWidth<=window.innerWidth`),'Mobile horizontal overflow');
 const mobile=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});writeFileSync(resolve(cache,'mobile.png'),Buffer.from(mobile.data,'base64'));
 assert.deepEqual(errors,[],'Browser runtime errors');assert.deepEqual(remote,[],'Remote requests');
 console.log('PASS: desktop/mobile, editing, operation buttons, all 5 drills, 7 examples, scalar reuse, zero remote page requests.');
 console.log('Screenshots: .cache/browser/{desktop,drills,mobile}.png');
}finally{if(socket)socket.close();chrome.kill('SIGTERM')}
