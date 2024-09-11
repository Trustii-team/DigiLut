from torchvision.transforms import v2
        
import torchstain
from PIL import Image
import numpy as np
from scipy.stats import scoreatpercentile

def normalize_staining(I, Io=240, beta=0.15, alpha=1, HERef=None, maxCRef=None):
    """
    Normalize the staining appearance of images originating from H&E stained sections.

    Mitkovetta. (n.d.). normalizeStaining.m. GitHub repository.
    Retrieved from https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m
    [1] A method for normalizing histology slides for quantitative analysis, M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun, C    Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
    This MATLAB code was translated to Python with assistance from ChatGPT, developed by OpenAI.

    Parameters:
        I (numpy.ndarray): RGB input image.
        Io (int, optional): Transmitted light intensity. Default is 240.
        beta (float, optional): OD threshold for transparent pixels. Default is 0.15.
        alpha (float, optional): Tolerance for the pseudo-min and pseudo-max. Default is 1.
        HERef (numpy.ndarray, optional): Reference H&E OD matrix. Default value is defined.
        maxCRef (numpy.ndarray, optional): Reference maximum stain concentrations for H&E. Default value is defined.

    Returns:
        Inorm (PIL.Image.Image): Normalized image.
        H (PIL.Image.Image, optional): Hematoxylin image.
        E (PIL.Image.Image, optional): Eosin image.

## How to call the function:
image_path = 'data/images/0w9NQUKyFU_b.tif'
slide = openslide.OpenSlide(image_path)

patch = show_image_patch(slide, 1900, 5600, 242, 242, 2)
patch = patch.convert("RGB")
Inorm, H, E = normalize_staining(patch)

###
Definitions:
	•	Normalized Image (Inorm): Appears similar to the original RGB image but with adjusted colors and intensity
        to match a standardized reference. It includes contributions from both Hematoxylin (blue/purple)
        and Eosin (pink/red) stains.
	•	Hematoxylin Image (H): Primarily shows the nuclei stained by Hematoxylin in shades of blue or purple.
        It lacks the pink/red hues contributed by Eosin.
    •	Eosin Image (E): The Eosin image is a single-channel image that represents only the Eosin stain component of the original image. Eosin typically stains the cytoplasm and extracellular matrix in various shades of pink or red.

    -> we are interested in either Inorm or H. Eosin is just the pink stuff without the 'cells'.
    I would suggest trying out the 'H' because it shows the cells which we are interested in.
    
##How to display the original and normalized images:
fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].imshow(image)
ax[0].set_title("Original Image")

ax[1].imshow(Inorm)
ax[1].set_title("Normalized Image")


if H is not None:
    ax[2].imshow(H)
    ax[2].set_title("Hematoxylin Image")
else:
    ax[2].imshow(image)
    ax[2].set_title("No Hematoxylin Image")

plt.show()
"""

    # Default reference H&E OD matrix
    if HERef is None:
        HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
    # Default reference maximum stain concentrations
    if maxCRef is None:
        maxCRef = np.array([1.9705, 1.0308])
    # Convert PIL image to numpy array if necessary
    if isinstance(I, Image.Image):
        I = np.array(I)
    
    h, w, c = I.shape
    I = I.reshape((-1, 3)).astype(float)
    
    # calculate optical density
    #OD = -np.log((I + 1) / Io)
    OD = -np.log10((I.astype(float)+1)/Io)
    
    # remove transparent pixels
    #ODhat = OD[np.all(OD >= beta, axis=1)]
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    #try:
    # calculate eigenvectors
    #_, V = np.linalg.eigh(np.cov(ODhat.T))
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    That = np.dot(ODhat, eigvecs[:, 1:3])
    
    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 1], That[:, 0])
    

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # Heuristic to make the vector corresponding to hematoxylin first and eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
        
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
        
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  

    # Separating H and E components

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    
    '''
    C = C / maxC[:, np.newaxis]
    C = C * maxCRef[:, np.newaxis]
    
    ###TODO change
    
    CLIP_LOWER = -709
    CLIP_UPPER = 709
    
    dot_product = np.dot(HERef, C)
    clipped_dot_product = np.clip(dot_product, CLIP_LOWER, CLIP_UPPER)
    
    # Recreate the image using reference mixing matrix
    #Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Io * np.exp(-clipped_dot_product)
    Inorm = Inorm.T.reshape(h, w, 3)
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
    Inorm = Image.fromarray(Inorm)
    
    H, E = None, None
    
    if C.shape[0] > 1:
        H = Io * np.exp(-np.outer(HERef[:, 0], C[0, :]))
        H = H.T.reshape(h, w, 3)
        H = np.clip(H, 0, 255).astype(np.uint8)
        H = Image.fromarray(H)
    
    if C.shape[0] > 1:
        E = Io * np.exp(-np.outer(HERef[:, 1], C[1, :]))
        E = E.T.reshape(h, w, 3)
        E = np.clip(E, 0, 255).astype(np.uint8)
        E = Image.fromarray(E)'''

    
    return Inorm, H, E


def choose_transform(transform_type, size, mean, std):
    if transform_type == 'train_transform':
        return _train_transform(size, mean, std)
    elif transform_type == 'train_transform2':
        return _train_transform2(size, mean, std)
    elif transform_type == 'val_transform':
        return _val_transform(size, mean, std)
    elif transform_type == 'test_transform':
        return _test_transform(size, mean, std)
    else:
        return NotImplementedError


def _train_transform(size, image_mean, image_std):
    
    
    return v2.RandomOrder(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),

            
        ]
    )
    
def _train_transform2(size, image_mean, image_std):
    
    
    return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomChoice(
        [   v2.RandomRotation(degrees=(0, 180), fill=255),
            v2.ColorJitter(contrast=0.2),
            v2.ColorJitter(brightness=0.2),
            v2.GaussianBlur(kernel_size=(3, 3)),
            v2.ElasticTransform(),
        ]
    ) ])

def _val_transform(size, image_mean, image_std):
    return v2.Compose(
        [ v2.Resize(size)
            #v2.ToTensor(),
        ]
    )

def _test_transform(size, image_mean, image_std):
    return v2.Compose(
        [ v2.Resize(size)

            #v2.ToTensor(),
        ]
    )
