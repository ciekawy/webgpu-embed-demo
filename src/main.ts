import { cos_sim } from '@xenova/transformers';
// import * as ort from 'onnxruntime-web';
import { computeEmbedding, loadExtractor } from './embeddings.ts';

// console.log('Hello from main.ts', ort);

interface Item {
    text: string;
    embedding: number[];
}

// let serviceWorker: ServiceWorker | null = null;

// async function registerServiceWorker() {
//     if ('serviceWorker' in navigator) {
//         try {
//             const registration = await navigator.serviceWorker.register('/src/sw.ts', { type: 'module' });
//             serviceWorker = registration.installing || registration.waiting || registration.active;
//             console.log('Service Worker registered successfully.');
//         } catch (error) {
//             console.error('Service Worker registration failed:', error);
//         }
//     }
// }
//
// async function computeEmbeddingViaWorker(text: string): Promise<number[] | null> {
//     if (!serviceWorker) {
//         console.error('Service Worker not available');
//         return null;
//     }
//
//     return new Promise((resolve) => {
//         const channel = new MessageChannel();
//         channel.port1.onmessage = (event) => {
//             resolve(event.data.embedding);
//         };
//         serviceWorker?.postMessage({ type: 'COMPUTE_EMBEDDING', text }, [channel.port2]);
//     });
// }

// @ts-ignore
const demoText = `
- Digital minimalism can enhance productivity by reducing distractions. Tim Ferriss advocates for focused work intervals, which can be particularly effective when combined with tools like the Pomodoro Technique.
- W podróży do Japonii warto odwiedzić Kioto, gdzie wiosną można podziwiać spektakularne kwitnienie wiśni, znane jako hanami.
- Cooking with fresh herbs like basil and thyme can elevate simple dishes; Ottolenghi's recipes often highlight the importance of these ingredients for flavor complexity.
- The concept of "Ikigai" from Japanese culture emphasizes finding a balance between passion, mission, vocation, and profession for a fulfilling life.
- Rozwój osobisty często wymaga konsekwentnego wprowadzania małych zmian, jak codzienne medytacje lub pisanie dziennika, co może przynieść długoterminowe korzyści.
- Integrating AI into healthcare can revolutionize patient care; for instance, IBM's Watson is being used to aid in diagnosing diseases and personalizing treatment plans.
- Culinary techniques like sous-vide allow for precise temperature control, ensuring perfectly cooked meats and vegetables every time.
- Exploring the history of ancient civilizations, such as the Inca Empire and its engineering marvels like Machu Picchu, provides insights into human innovation.
- Zachowanie równowagi między życiem zawodowym a prywatnym może być wspomagane przez techniki takie jak mindfulness, promowane przez autorów takich jak Jon Kabat-Zinn.
- The rapid development of quantum computing, led by companies like Google and IBM, promises to solve complex problems that are currently intractable with classical computers.
- Baking bread at home can be both therapeutic and rewarding; using sourdough starter from Tartine Bakery's method brings out complex flavors and textures.
- Visiting Iceland offers unique experiences such as bathing in the Blue Lagoon and witnessing the Northern Lights, creating unforgettable memories.
- Samorealizacja może być osiągnięta przez ciągłe stawianie sobie wyzwań i naukę nowych umiejętności, jak zaleca Tony Robbins w swoich seminariach.
- Renewable energy technologies, like Tesla's solar roof tiles, are making sustainable living more accessible and practical for homeowners.
- In Italian cuisine, the simplicity of dishes like cacio e pepe highlights the importance of quality ingredients and traditional techniques.
- Podróżując po Włoszech, warto odwiedzić mniej znane regiony, takie jak Puglia, gdzie można odkryć autentyczną kulturę i kuchnię z dala od turystycznych szlaków.
- Emotional intelligence, a concept popularized by Daniel Goleman, is crucial for building strong personal and professional relationships.
- Blockchain technology, beyond cryptocurrencies, has the potential to revolutionize industries by providing secure, transparent transaction methods.
- Fermentacja warzyw, np. kapusty na kiszonki, to metoda konserwacji jedzenia, która jednocześnie wzbogaca je o probiotyki, korzystne dla zdrowia jelit.
- Experiencing local festivals, like the Holi festival in India, offers a deep dive into the cultural and social fabric of a country.
- Personal branding, as suggested by experts like Gary Vaynerchuk, is essential in the digital age for career advancement and networking.
- Advances in biotechnology, such as CRISPR gene editing, open new possibilities for treating genetic disorders and improving crop resilience.
- Przygotowanie tradycyjnych dań, takich jak polskie pierogi, może być doskonałą okazją do integracji rodzinnej i przekazywania kulinarnych tradycji.
- Traveling through Southeast Asia provides a diverse culinary adventure, from street food in Bangkok to fine dining in Singapore.
- Practicing gratitude daily, as recommended by positive psychology researchers like Martin Seligman, can significantly enhance overall well-being.
- The Internet of Things (IoT) is transforming homes into smart environments, with devices like Amazon's Alexa providing seamless integration and control.
- Uprawianie jogi, według wskazówek takich jak te od Adriene Mishler, może pomóc w redukcji stresu i poprawie elastyczności ciała.
- Exploring the wine regions of France, such as Bordeaux and Burgundy, offers not only tasting experiences but also lessons in history and terroir.
- Networking, both online and offline, is vital for professional growth; platforms like LinkedIn offer opportunities for global connections.
- Sustainable fashion, promoted by brands like Patagonia, emphasizes the importance of ethical production and reducing environmental impact.
- Mindfulness meditation, as taught by Thich Nhat Hanh, can help cultivate a sense of peace and presence in everyday life.
- Przeglądanie starych książek kucharskich może być inspiracją do odkrywania zapomnianych przepisów i technik kulinarnych.
- Exploring the Arctic region offers a unique perspective on climate change, as seen through the melting glaciers and shifting ecosystems.
- The philosophy of Stoicism, with texts by Marcus Aurelius and Seneca, provides practical wisdom for handling life's challenges.
- Gotowanie z użyciem lokalnych składników, takich jak w kuchni farm-to-table, promowanej przez restauracje jak Blue Hill, wspiera lokalne rolnictwo i zrównoważony rozwój.
- Experiencing the vibrant street art scene in cities like Berlin can reveal a city's social and political narratives.
- Developing a growth mindset, as described by Carol Dweck, encourages resilience and a love of learning.
- Wizyta w małych winnicach w regionie Toskanii pozwala na odkrycie unikalnych smaków i historii związanych z produkcją wina.
- The rise of telemedicine, accelerated by the COVID-19 pandemic, is making healthcare more accessible and convenient.
- Cooking with seasonal ingredients, as highlighted in Alice Waters' cookbooks, brings out the best flavors and supports local farmers.
- Exploring the architectural wonders of Antoni Gaudí in Barcelona offers a glimpse into innovative and organic design.
- Regular physical activity, like the high-intensity interval training (HIIT) popularized by Joe Wicks, is crucial for maintaining health and fitness.
- Fermentowanie własnych napojów, jak kombucha, może być fascynującym hobby, a także źródłem zdrowych probiotyków.
- The concept of slow travel emphasizes immersing oneself in a destination rather than rushing through tourist spots.
- Developing critical thinking skills, as advocated by Edward de Bono, is essential for problem-solving and innovation.
- In the tech world, understanding blockchain's potential beyond cryptocurrency, such as in supply chain management, is becoming increasingly important.
- Odkrywanie regionalnych kuchni, jak kuchnia Meksyku z jej bogatą paletą smaków i technik, może być kulinarną podróżą do innego świata.
- Practicing minimalism, inspired by Marie Kondo, can lead to a more organized and fulfilling life by focusing on what truly matters.
- Renewable energy solutions, like offshore wind farms developed by companies such as Ørsted, are crucial for combating climate change.
- Experimenting with fusion cuisine, combining elements from different culinary traditions, can result in innovative and exciting dishes.
- Learning a new language, such as using apps like Duolingo, can open doors to different cultures and enhance cognitive abilities.
- The trend of urban farming, supported by initiatives like rooftop gardens, promotes sustainability and fresh produce in city environments.
- Discovering hidden gems in travel, like the quaint villages of the Cotswolds, offers a peaceful retreat from bustling tourist spots.
- Practicing empathy, as suggested by Brene Brown, strengthens personal relationships and fosters a supportive community.
- Advances in artificial intelligence, particularly in natural language processing, are revolutionizing how we interact with technology.
- Robienie własnych kosmetyków, jak balsamy i mydła, pozwala na kontrolowanie składników i dbanie o skórę w naturalny sposób.
- The rise of digital nomadism, facilitated by remote work technology, allows for a lifestyle combining work and travel.
- Developing a habit of daily reading, inspired by leaders like Warren Buffett, can significantly broaden one's knowledge and perspective.
- Exploring national parks, such as Yellowstone and Yosemite, provides a connection to nature and an appreciation for conservation efforts.
- Learning about permaculture, a sustainable agricultural practice promoted by Bill Mollison, can inspire eco-friendly gardening techniques.
`

const items: Item[] = loadItemsFromLocalStorage() || [];

async function processBulkInput(demo = false) {
    if (demo) {
        items.push(...await fetch('demo-data.json').then(response => response.json()));
    }
    const bulkInputElement = document.getElementById('bulk-input') as HTMLTextAreaElement;
    const bulkText = bulkInputElement.value;
    const processedItems = filterAndProcessLines(bulkText.split('\n'));
    bulkInputElement.disabled = true;
    await addProcessedItems(processedItems);
    bulkInputElement.disabled = false;
}

function filterAndProcessLines(lines: string[]): string[] {
    const filteredLines = lines.filter(line => /\w{3}/.test(line));

    const processedLines: string[] = [];
    let buffer = "";

    for (const line of filteredLines) {
        if (/^\s*-\s*/.test(line) || /^\s*$/.test(line)) {
            if (buffer) {
                processedLines.push(buffer.trim());
                buffer = "";
            }
            if (/^\s*-\s*/.test(line)) {
                buffer = line.replace(/^\s*-\s*/, '').trim();
            }
        } else {
            buffer += ' ' + line.trim();
        }
    }

    if (buffer) {
        processedLines.push(buffer.trim());
    }

    return processedLines;
}

async function addProcessedItems(processedItems: string[]) {
    // const extractor = await loadExtractor();
    for (const item of processedItems) {
        const embedding = await computeEmbedding(item);
        if (embedding) {
            items.push({ text: item, embedding });
        }
    }
    saveItemsToLocalStorage();
    updateItemList();
}


async function addItem() {
    const newItemInput = document.getElementById('new-item') as HTMLInputElement;
    const newItem = newItemInput.value;
    if (newItem && newItem.length >= 3) {
        const embedding = await computeEmbedding(newItem);
        if (embedding) {
            items.push({ text: newItem, embedding });
            saveItemsToLocalStorage();
            updateItemList();
        }
    }
}

function updateItemList() {
    const itemList = document.getElementById('item-list') as HTMLUListElement;
    itemList.innerHTML = '';
    items.forEach((item, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            ${item.text} 
            <a href="#" class="delete-link" data-index="${index}">[Delete]</a>
        `;
        itemList.appendChild(li);
    });

    // Add event listeners to delete links
    const deleteLinks = document.querySelectorAll('.delete-link');
    deleteLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const index = parseInt((e.target as HTMLAnchorElement).getAttribute('data-index') || '-1');
            if (index !== -1) {
                deleteItem(index);
            }
        });
    });
}

function deleteItem(index: number) {
    items.splice(index, 1);
    saveItemsToLocalStorage();
    updateItemList();
    // Optionally, update similar items if you want
    findSimilarItems();
}

function saveItemsToLocalStorage() {
    localStorage.setItem('items', JSON.stringify(items));
}

function loadItemsFromLocalStorage(): Item[] | null {
    const itemsJson = localStorage.getItem('items');
    return itemsJson ? JSON.parse(itemsJson) : null;
}


function resetItems() {
    localStorage.removeItem('items');
    items.length = 0;
    updateItemList();
    const similarItemsList = document.getElementById('similar-items') as HTMLUListElement;
    similarItemsList.innerHTML = '';
}

const findSimilarItems = debounce(async () => {
    const newItemInput = document.getElementById('new-item') as HTMLInputElement;
    const newItem = newItemInput.value;
    if (newItem.length >= 3 && items.length > 0) {
        const newEmbedding = await computeEmbedding(`notatka na temat: ${newItem}`);
        if (newEmbedding) {
            const similarities = items.map(item => ({
                text: item.text,
                similarity: cos_sim(newEmbedding, item.embedding)
            }));
            similarities.sort((a, b) => b.similarity - a.similarity);
            updateSimilarItems(similarities.slice(0, 10));
        }
    } else {
        console.log('No items to compare with.');
    }
}, 300);

function updateSimilarItems(similarItems: { text: string; similarity: number }[]) {
    const similarItemsList = document.getElementById('similar-items') as HTMLUListElement;
    similarItemsList.innerHTML = '';
    similarItems.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `(S: ${item.similarity.toFixed(3)}) ${item.text}`;
        similarItemsList.appendChild(li);
    });
}

function debounce(func: (...args: any[]) => void, wait: number) {
    let timeout: number | undefined;
    return function(...args: any[]) {
        clearTimeout(timeout);
        // @ts-ignore
        timeout = window.setTimeout(() => func.apply(this, args), wait);
    };
}

function updateProgressBar({ progress }: { progress: number }) {
    console.log('Progress:', progress);
  document.getElementById('progress-bar')!.style.width = `${progress}%`;
}

function completeProgress() {
  updateProgressBar({ progress: 100 });
  setTimeout(() => {
    document.getElementById('progress-container')!.style.display = 'none';
  }, 500); // Hide the progress bar after a short delay
}

// async function loadModel() {
//   const modelUrl = 'https://example.com/path-to-your-model.onnx';
//
//   const onProgress = (loaded: number, total: number) => {
//     const progress = (loaded / total) * 100;
//     updateProgressBar(progress);
//   };
//
//   try {
//     const response = await fetchWithProgress(modelUrl, onProgress);
//     // @ts-ignore
//     const modelData = await response.arrayBuffer();
//     // Assuming this is where the model is initialized
//     // Example: await ort.InferenceSession.create(modelData);
//
//     // Simulate model initialization
//     await new Promise(resolve => setTimeout(resolve, 2000)); // Simulating model initialization
//     completeProgress();
//   } catch (error) {
//     console.error('Error fetching the model:', error);
//   }
// }


if (window !== undefined) {


    window.onload = async () => {
        document.getElementById('main')!.style.display = 'none';
        // await registerServiceWorker();
        // await loadModel();
        await loadExtractor(updateProgressBar);
        completeProgress();
        document.getElementById('main')!.style.display = 'initial';
        updateItemList();
    };

    // Assign functions to the window object
    (window as any).addItem = addItem;
    (window as any).resetItems = resetItems;
    (window as any).findSimilarItems = findSimilarItems;
    (window as any).deleteItem = deleteItem;
    (window as any).processBulkInput = processBulkInput;
}
