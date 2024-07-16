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


const items: Item[] = loadItemsFromLocalStorage() || [];

async function processBulkInput() {
    const bulkInputElement = document.getElementById('bulk-input') as HTMLTextAreaElement;
    const bulkText = bulkInputElement.value;
    const processedItems = filterAndProcessLines(bulkText.split('\n'));
    await addProcessedItems(processedItems);
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

if (window !== undefined) {
    window.onload = async () => {
        // await registerServiceWorker();
        await loadExtractor();
        updateItemList();
    };

    // Assign functions to the window object
    (window as any).addItem = addItem;
    (window as any).resetItems = resetItems;
    (window as any).findSimilarItems = findSimilarItems;
    (window as any).deleteItem = deleteItem;
    (window as any).processBulkInput = processBulkInput;
}
